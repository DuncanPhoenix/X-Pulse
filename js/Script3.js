
import * as THREE from "https://esm.sh/three@0.175.0";
import { GUI } from "https://esm.sh/dat.gui@0.7.9";
import Stats from "https://esm.sh/stats.js@0.17.0";

// Create renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Create render targets for feedback
const rtParams = {
  minFilter: THREE.LinearFilter,
  magFilter: THREE.LinearFilter,
  format: THREE.RGBAFormat,
  type: THREE.FloatType
};

const renderTarget1 = new THREE.WebGLRenderTarget(
  window.innerWidth,
  window.innerHeight,
  rtParams
);

const renderTarget2 = renderTarget1.clone();

// Create scenes and cameras
const simulationScene = new THREE.Scene();
const renderScene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

// Parameters for GUI
const params = {
  // Simulation type
  simulationType: "fluidDynamics", // Default simulation

  // Common parameters
  forceIntensity: 1.0,
  decayRate: 0.99,
  feedbackStrength: 0.8,

  // Fluid dynamics parameters
  viscosity: 0.85,
  velocityScale: 1.0,
  pressureIterations: 16,

  // Curl noise flow parameters
  noiseScale: 0.8,
  noiseSpeed: 0.2,
  noiseOctaves: 4,
  noiseTwist: 1.2,

  // Color settings
  primaryColor: [0.02, 0.1, 0.4],
  accentColor: [1.0, 1.0, 1.0],
  colorIntensity: 5.0,
  colorSaturation: 1.0,

  // Auto-movement
  autoMove: true,
  autoMoveWhenInactive: true,
  movementRadius: 0.6,
  movementSpeed: 0.2,

  // Presets
  preset: "ocean"
};

// Color presets
const presets = {
  ocean: {
    primaryColor: [0.02, 0.1, 0.4],
    accentColor: [1.0, 1.0, 1.0],
    colorIntensity: 5.0,
    colorSaturation: 1.0
  },
  fire: {
    primaryColor: [0.4, 0.05, 0.01],
    accentColor: [1.0, 0.7, 0.3],
    colorIntensity: 4.0,
    colorSaturation: 1.2
  },
  toxic: {
    primaryColor: [0.05, 0.3, 0.05],
    accentColor: [0.8, 1.0, 0.2],
    colorIntensity: 4.5,
    colorSaturation: 1.3
  },
  neon: {
    primaryColor: [0.2, 0.05, 0.3],
    accentColor: [0.9, 0.4, 1.0],
    colorIntensity: 3.5,
    colorSaturation: 1.5
  }
};

// Apply a preset
function applyPreset(presetName) {
  const preset = presets[presetName];
  if (!preset) return;

  params.primaryColor = [...preset.primaryColor];
  params.accentColor = [...preset.accentColor];
  params.colorIntensity = preset.colorIntensity;
  params.colorSaturation = preset.colorSaturation;
  params.preset = presetName;

  // Update uniforms
  renderMaterial.uniforms.primaryColor.value.set(...params.primaryColor);
  renderMaterial.uniforms.accentColor.value.set(...params.accentColor);
  renderMaterial.uniforms.colorIntensity.value = params.colorIntensity;
  renderMaterial.uniforms.colorSaturation.value = params.colorSaturation;

  // Update GUI
  for (const controller of colorFolder.__controllers) {
    controller.updateDisplay();
  }
}

// Mouse tracking
const mouse = new THREE.Vector2(0, 0);
let mouseActive = false;
let lastMouseMoveTime = 0;

window.addEventListener("mousemove", (event) => {
  mouse.x = event.clientX;
  mouse.y = window.innerHeight - event.clientY;
  mouseActive = true;
  lastMouseMoveTime = performance.now();
});

window.addEventListener("mouseout", () => {
  mouseActive = false;
});

// Initialize with some data
const initTexture = function () {
  const size = 256;
  const data = new Float32Array(4 * size * size);

  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const idx = (i * size + j) * 4;
      data[idx] = 0; // R - x velocity
      data[idx + 1] = 0; // G - y velocity
      data[idx + 2] = 0.05; // B - fluid density
      data[idx + 3] = 0; // A - pressure
    }
  }

  const texture = new THREE.DataTexture(
    data,
    size,
    size,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  texture.needsUpdate = true;
  return texture;
};

// Common utilities for simulations
const shaderUtils = `
      // Random functions
      float hash(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
      }
      
      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        
        float a = hash(i);
        float b = hash(i + vec2(1.0, 0.0));
        float c = hash(i + vec2(0.0, 1.0));
        float d = hash(i + vec2(1.0, 1.0));
        
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
      }
      
      // Simplex 3D noise
      vec3 hash3(vec3 p) {
        p = vec3(
          dot(p, vec3(127.1, 311.7, 74.7)),
          dot(p, vec3(269.5, 183.3, 246.1)),
          dot(p, vec3(113.5, 271.9, 124.6))
        );
        return fract(sin(p) * 43758.5453123);
      }
      
      float simplex3d(vec3 p) {
        const float K1 = 0.333333333;
        const float K2 = 0.166666667;
        
        vec3 i = floor(p + (p.x + p.y + p.z) * K1);
        vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
        
        vec3 e = step(vec3(0.0), d0 - d0.yzx);
        vec3 i1 = e * (1.0 - e.zxy);
        vec3 i2 = 1.0 - e.zxy * (1.0 - e);
        
        vec3 d1 = d0 - (i1 - K2);
        vec3 d2 = d0 - (i2 - K1 * 2.0);
        vec3 d3 = d0 - (1.0 - 3.0 * K2);
        
        vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
        vec4 n = h * h * h * h * vec4(
          dot(d0, hash3(i)),
          dot(d1, hash3(i + i1)),
          dot(d2, hash3(i + i2)),
          dot(d3, hash3(i + 1.0))
        );
        
        return dot(vec4(31.316), n);
      }
      
      vec2 getForceCenter(bool useAutoMovement, vec4 iMouse, vec2 iResolution, float iTime, float movementRadius, float movementSpeed) {
        vec2 forceCenter;
        
        if (useAutoMovement) {
          float r = 0.2 + movementRadius * sin(iTime * 0.5);
          forceCenter = vec2(
            0.5 + r * 0.5 * sin(iTime * movementSpeed),
            0.5 + r * 0.5 * cos(iTime * movementSpeed)
          );
        } else {
          forceCenter = iMouse.xy / iResolution.xy;
        }
        
        return forceCenter;
      }
    `;

/* 1. FLUID DYNAMICS SIMULATION */
const fluidDynamicsMaterial = new THREE.ShaderMaterial({
  uniforms: {
    iFrame: { value: 0 },
    iResolution: {
      value: new THREE.Vector2(window.innerWidth, window.innerHeight)
    },
    iTime: { value: 0 },
    iMouse: { value: new THREE.Vector4(0, 0, 0, 0) },
    iChannel0: { value: null },

    // Specific parameters
    viscosity: { value: params.viscosity },
    velocityScale: { value: params.velocityScale },
    pressureIterations: { value: params.pressureIterations },

    // Common parameters
    forceIntensity: { value: params.forceIntensity },
    decayRate: { value: params.decayRate },
    feedbackStrength: { value: params.feedbackStrength },

    // Auto movement
    autoMove: { value: params.autoMove },
    autoMoveWhenInactive: { value: params.autoMoveWhenInactive },
    movementRadius: { value: params.movementRadius },
    movementSpeed: { value: params.movementSpeed }
  },
  vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position, 1.0);
        }
      `,
  fragmentShader: `
        uniform int iFrame;
        uniform vec2 iResolution;
        uniform float iTime;
        uniform vec4 iMouse;
        uniform sampler2D iChannel0;
        
        // Specific parameters
        uniform float viscosity;
        uniform float velocityScale;
        uniform int pressureIterations;
        
        // Common parameters
        uniform float forceIntensity;
        uniform float decayRate;
        uniform float feedbackStrength;
        
        // Auto movement
        uniform bool autoMove;
        uniform bool autoMoveWhenInactive;
        uniform float movementRadius;
        uniform float movementSpeed;
        
        varying vec2 vUv;
        
        ${shaderUtils}
        
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
          vec2 uv = fragCoord.xy / iResolution.xy;
          vec2 pixelSize = 1.0 / iResolution.xy;
          
          // In this simulation:
          // R, G channels: Velocity (x, y)
          // B channel: Density
          // A channel: Pressure
          
          // Get current state
          vec4 data = texture(iChannel0, uv);
          vec2 velocity = data.xy;
          float density = data.z;
          float pressure = data.w;
          
          // Calculate force center position (mouse or auto-movement)
          bool useAutoMovement = autoMove && (iMouse.z <= 0.5 || !autoMoveWhenInactive);
          vec2 forceCenter = getForceCenter(useAutoMovement, iMouse, iResolution, iTime, movementRadius, movementSpeed);
          
          // Add forces from input
          vec2 toCenter = forceCenter - uv;
          float distToCenter = length(toCenter);
          
          // Apply forces based on distance to mouse/force center
          if (distToCenter < 0.1) {
            float forceFactor = (0.1 - distToCenter) * 10.0 * forceIntensity;
            
            // Swirling force - perpendicular to radial direction with some radial component
            vec2 tangent = vec2(-toCenter.y, toCenter.x);
            vec2 swirl = mix(normalize(tangent), normalize(toCenter), 0.3);
            
            // Apply force with pulsating effect
            float pulse = 0.5 + 0.5 * sin(iTime * 3.0);
            velocity += swirl * forceFactor * pulse * velocityScale * 0.01;
            
            // Add density at force center
            density += forceFactor * 0.1 * pulse;
          }
          
          // Advection step - sample velocity from previous position
          vec2 prevPos = uv - velocity * velocityScale * pixelSize.x * 10.0;
          
          // Ensure sampling within bounds
          if (prevPos.x >= 0.0 && prevPos.x <= 1.0 && prevPos.y >= 0.0 && prevPos.y <= 1.0) {
            vec4 prevData = texture(iChannel0, prevPos);
            
            // Apply advection with damping (viscosity)
            velocity = mix(velocity, prevData.xy, viscosity);
            
            // Advect density
            density = mix(density, prevData.z, feedbackStrength);
          }
          
          // Diffusion through simplified Laplacian
          vec2 laplacianV = vec2(0.0);
          float laplacianD = 0.0;
          
          // Sample neighbors
          vec4 n1 = texture(iChannel0, uv + vec2(pixelSize.x, 0.0));
          vec4 n2 = texture(iChannel0, uv - vec2(pixelSize.x, 0.0));
          vec4 n3 = texture(iChannel0, uv + vec2(0.0, pixelSize.y));
          vec4 n4 = texture(iChannel0, uv - vec2(0.0, pixelSize.y));
          
          // Compute Laplacian (∇²)
          laplacianV = n1.xy + n2.xy + n3.xy + n4.xy - 4.0 * velocity;
          laplacianD = n1.z + n2.z + n3.z + n4.z - 4.0 * density;
          
          // Apply diffusion
          velocity += laplacianV * 0.05 * (1.0 - viscosity);
          density += laplacianD * 0.01;
          
          // Compute divergence for pressure projection
          float divergence = 
            (n1.x - n2.x) / (2.0 * pixelSize.x) + 
            (n3.y - n4.y) / (2.0 * pixelSize.y);
          
          // Simplified pressure solver
          float newPressure = (n1.w + n2.w + n3.w + n4.w - divergence) * 0.25;
          
          // Apply pressure gradient to enforce incompressibility
          vec2 pressureGradient;
          pressureGradient.x = (n1.w - n2.w) / (2.0 * pixelSize.x);
          pressureGradient.y = (n3.w - n4.w) / (2.0 * pixelSize.y);
          
          // Update velocity by subtracting pressure gradient
          velocity -= pressureGradient * 0.1;
          
          // Boundary conditions - dampen velocity at borders
          float borderWidth = 0.05;
          if (uv.x < borderWidth || uv.x > 1.0 - borderWidth || 
              uv.y < borderWidth || uv.y > 1.0 - borderWidth) {
            velocity *= 0.9;
          }
          
          // Apply decay
          velocity *= decayRate;
          density *= decayRate;
          
          // Limit velocity for stability
          float maxVel = 0.1 * velocityScale;
          if (length(velocity) > maxVel) {
            velocity = normalize(velocity) * maxVel;
          }
          
          // Ensure density stays in valid range
          density = clamp(density, 0.0, 1.0);
          
          // Output final state
          fragColor = vec4(velocity, density, newPressure);
        }
        
        void main() {
          mainImage(gl_FragColor, vUv * iResolution.xy);
        }
      `
});

/* 2. CURL NOISE FLOW SIMULATION */
const curlNoiseFlowMaterial = new THREE.ShaderMaterial({
  uniforms: {
    iFrame: { value: 0 },
    iResolution: {
      value: new THREE.Vector2(window.innerWidth, window.innerHeight)
    },
    iTime: { value: 0 },
    iMouse: { value: new THREE.Vector4(0, 0, 0, 0) },
    iChannel0: { value: null },

    // Specific parameters
    noiseScale: { value: params.noiseScale },
    noiseSpeed: { value: params.noiseSpeed },
    noiseOctaves: { value: params.noiseOctaves },
    noiseTwist: { value: params.noiseTwist },

    // Common parameters
    forceIntensity: { value: params.forceIntensity },
    decayRate: { value: params.decayRate },
    feedbackStrength: { value: params.feedbackStrength },

    // Auto movement
    autoMove: { value: params.autoMove },
    autoMoveWhenInactive: { value: params.autoMoveWhenInactive },
    movementRadius: { value: params.movementRadius },
    movementSpeed: { value: params.movementSpeed }
  },
  vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position, 1.0);
        }
      `,
  fragmentShader: `
        uniform int iFrame;
        uniform vec2 iResolution;
        uniform float iTime;
        uniform vec4 iMouse;
        uniform sampler2D iChannel0;
        
        // Specific parameters
        uniform float noiseScale;
        uniform float noiseSpeed;
        uniform int noiseOctaves;
        uniform float noiseTwist;
        
        // Common parameters
        uniform float forceIntensity;
        uniform float decayRate;
        uniform float feedbackStrength;
        
        // Auto movement
        uniform bool autoMove;
        uniform bool autoMoveWhenInactive;
        uniform float movementRadius;
        uniform float movementSpeed;
        
        varying vec2 vUv;
        
        ${shaderUtils}
        
        // Calculate curl of 3D noise field to get divergence-free velocity field
        vec2 curlNoise(vec2 p, float t) {
          float epsilon = 0.001;
          
          // Calculate noise gradient in z direction at two close points
          float n1 = simplex3d(vec3(p, t));
          float n2 = simplex3d(vec3(p + vec2(epsilon, 0.0), t));
          float n3 = simplex3d(vec3(p + vec2(0.0, epsilon), t));
          
          // Approximate partial derivatives
          float dx = (n2 - n1) / epsilon;
          float dy = (n3 - n1) / epsilon;
          
          // Return curl (perpendicular to gradient)
          return vec2(-dy, dx);
        }
        
        vec2 flowField(vec2 p, float t) {
          vec2 flow = vec2(0.0);
          float amplitude = 1.0;
          float frequency = 1.0;
          
          // Fractal Brownian Motion (FBM) for curl noise
          for (int i = 0; i < 8; i++) {
            if (i >= noiseOctaves) break;
            
            // Calculate curl noise at this octave
            vec2 curl = curlNoise(p * frequency * noiseScale, t * noiseSpeed);
            
            // Add to flow with decreasing amplitude for higher frequencies
            flow += curl * amplitude;
            
            // Adjust for next octave
            amplitude *= 0.5;
            frequency *= 2.0;
            
            // Apply twist for more interesting flow
            p += vec2(-p.y, p.x) * noiseTwist * 0.01;
          }
          
          return flow;
        }
        
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
          vec2 uv = fragCoord.xy / iResolution.xy;
          
          // In this simulation:
          // R, G channels: Velocity (x, y)
          // B channel: Density
          // A channel: Vorticity
          
          // Get current state
          vec4 data = texture(iChannel0, uv);
          vec2 velocity = data.xy;
          float density = data.z;
          float vorticity = data.w;
          
          // Calculate position in normalized space (-1 to 1)
          vec2 p = (uv * 2.0 - 1.0);
          p.x *= iResolution.x / iResolution.y; // Correct for aspect ratio
          
          // Calculate force center position (mouse or auto-movement)
          bool useAutoMovement = autoMove && (iMouse.z <= 0.5 || !autoMoveWhenInactive);
          vec2 forceCenter = getForceCenter(useAutoMovement, iMouse, iResolution, iTime, movementRadius, movementSpeed);
          
          // Calculate force center in normalized space
          vec2 fc = (forceCenter * 2.0 - 1.0);
          fc.x *= iResolution.x / iResolution.y;
          
          // Get base flow field from curl noise
          vec2 flow = flowField(p, iTime);
          
          // Add influence from force center
          vec2 toForce = fc - p;
          float distToForce = length(toForce);
          
          if (distToForce < 0.5) {
            // Strength falls off with distance
            float strength = (0.5 - distToForce) * 2.0 * forceIntensity;
            
            // Create swirling forces around force center
            vec2 perpendicular = vec2(-toForce.y, toForce.x);
            vec2 forceField = mix(normalize(perpendicular), normalize(toForce), 0.2);
            
            // Add forces to flow
            flow += forceField * strength * 0.1;
          }
          
          // Scale flow to get velocity
          vec2 newVelocity = flow * 0.01;
          
          // Smooth velocity changes
          velocity = mix(velocity, newVelocity, 0.1);
          
          // Apply advection to create fluid-like behavior
          vec2 prevPos = uv - velocity;
          
          // Sample previous frame with bilinear filtering
          if (prevPos.x >= 0.0 && prevPos.x <= 1.0 && prevPos.y >= 0.0 && prevPos.y <= 1.0) {
            vec4 prevData = texture(iChannel0, prevPos);
            
            // Create fluid-like feedback effect
            density = mix(density, prevData.z, feedbackStrength);
          }
          
          // Calculate vorticity (curl of velocity field)
          float eps = 0.001;
          float curl = 
            (texture(iChannel0, uv + vec2(eps, 0.0)).y - texture(iChannel0, uv - vec2(eps, 0.0)).y) -
            (texture(iChannel0, uv + vec2(0.0, eps)).x - texture(iChannel0, uv - vec2(0.0, eps)).x);
          
          // Smooth vorticity over time
          vorticity = mix(vorticity, curl, 0.1);
          
          // Add density based on vorticity and force center
          float vortexDensity = abs(vorticity) * 2.0;
          float forceDensity = 1.0 / (distToForce * 10.0 + 1.0);
          
          density += vortexDensity * 0.01 + forceDensity * 0.02;
          
          // Limit density
          density = min(density, 1.0);
          
          // Apply decay
          density *= decayRate;
          
          // Output final state
          fragColor = vec4(velocity, density, vorticity);
        }
        
        void main() {
          mainImage(gl_FragColor, vUv * iResolution.xy);
        }
      `
});

// Common renderer for all simulations
const renderMaterial = new THREE.ShaderMaterial({
  uniforms: {
    iResolution: {
      value: new THREE.Vector2(window.innerWidth, window.innerHeight)
    },
    iTime: { value: 0 },
    iChannel0: { value: null },

    // Color parameters
    primaryColor: { value: new THREE.Vector3(...params.primaryColor) },
    accentColor: { value: new THREE.Vector3(...params.accentColor) },
    colorIntensity: { value: params.colorIntensity },
    colorSaturation: { value: params.colorSaturation }
  },
  vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position, 1.0);
        }
      `,
  fragmentShader: `
        uniform vec2 iResolution;
        uniform float iTime;
        uniform sampler2D iChannel0;
        
        // Color parameters
        uniform vec3 primaryColor;
        uniform vec3 accentColor;
        uniform float colorIntensity;
        uniform float colorSaturation;
        
        varying vec2 vUv;
        
        // HSV to RGB conversion for color manipulation
        vec3 hsv2rgb(vec3 c) {
          vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
          vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
          return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
          vec2 uv = fragCoord.xy / iResolution.xy;
          
          // Get simulation data
          vec4 data = texture(iChannel0, uv);
          
          // Extract density from the simulation (always in blue channel)
          float density = data.z;
          
          // Apply fluid-like appearance to any simulation type
          vec3 color = primaryColor * (density * colorIntensity);
          
          // Add accents based on velocity/extra data
          float extraData = length(data.xy) * 2.0; // Using velocity magnitude or other data
          color += accentColor * extraData * density;
          
          // Add some variation based on time
          float timeEffect = 0.5 + 0.5 * sin(iTime * 0.2 + uv.x * 10.0 + uv.y * 8.0);
          color = mix(color, color * vec3(0.9, 1.0, 1.1), timeEffect * 0.2);
          
          // Add some subtle normal-map-like effects
          vec2 eps = 1.0 / iResolution.xy;
          float h1 = texture(iChannel0, uv + vec2(eps.x, 0.0)).z;
          float h2 = texture(iChannel0, uv - vec2(eps.x, 0.0)).z;
          float h3 = texture(iChannel0, uv + vec2(0.0, eps.y)).z;
          float h4 = texture(iChannel0, uv - vec2(0.0, eps.y)).z;
          
          vec3 normal = normalize(vec3(h2 - h1, h4 - h3, 0.01));
          float lighting = 0.5 + 0.5 * dot(normal, normalize(vec3(1.0, 1.0, 1.0)));
          
          color *= mix(0.8, 1.2, lighting);
          
          // Add glow for brighter areas
          float glow = pow(density, 2.0) * 0.5;
          color += accentColor * glow;
          
          // Adjust saturation
          float luminance = dot(color, vec3(0.299, 0.587, 0.114));
          color = mix(vec3(luminance), color, colorSaturation);
          
          // Output final color
          fragColor = vec4(color, 1.0);
        }
        
        void main() {
          mainImage(gl_FragColor, vUv * iResolution.xy);
        }
      `
});

// Create quads for simulation and rendering
const simulationQuad = new THREE.Mesh(
  new THREE.PlaneGeometry(2, 2),
  fluidDynamicsMaterial // Default simulation
);
simulationScene.add(simulationQuad);

const renderQuad = new THREE.Mesh(
  new THREE.PlaneGeometry(2, 2),
  renderMaterial
);
renderScene.add(renderQuad);

// Add stats
const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

// Function to update the simulation type
function updateSimulationType(simType) {
  // Remove current material
  simulationScene.remove(simulationQuad);

  // Update material based on simulation type
  switch (simType) {
    case "fluidDynamics":
      simulationQuad.material = fluidDynamicsMaterial;
      break;
    case "curlNoiseFlow":
      simulationQuad.material = curlNoiseFlowMaterial;
      break;
  }

  // Add with new material
  simulationScene.add(simulationQuad);

  // Reset simulation state
  frameCount = 0;

  // Clear render targets
  renderer.setRenderTarget(renderTarget1);
  renderer.clear();
  renderer.setRenderTarget(renderTarget2);
  renderer.clear();

  // Reset with initial texture
  const initialTexture = initTexture();
  simulationQuad.material.uniforms.iChannel0.value = initialTexture;
  renderer.setRenderTarget(previousTarget);
  renderer.render(simulationScene, camera);

  // Update params
  params.simulationType = simType;

  // Show relevant folders
  commonFolder.open();
  colorFolder.open();
  movementFolder.open();

  if (simType === "fluidDynamics") {
    fluidFolder.open();
    curlFolder.close();
  } else {
    fluidFolder.close();
    curlFolder.open();
  }
}

// Create GUI
const gui = new GUI();

// Simulation type selector
const typeFolder = gui.addFolder("Simulation Type");
typeFolder
  .add(params, "simulationType", {
    "Fluid Dynamics": "fluidDynamics",
    "Curl Noise Flow": "curlNoiseFlow"
  })
  .onChange(updateSimulationType);
typeFolder.open();

// Color presets
const presetFolder = gui.addFolder("Color Presets");
presetFolder
  .add(params, "preset", {
    "Ocean Blue": "ocean",
    "Fire Red": "fire",
    "Toxic Green": "toxic",
    "Neon Purple": "neon"
  })
  .onChange((value) => {
    applyPreset(value);
  });
presetFolder.open();

// Common parameters
const commonFolder = gui.addFolder("Common Parameters");
commonFolder.add(params, "forceIntensity", 0.1, 5.0).onChange((value) => {
  fluidDynamicsMaterial.uniforms.forceIntensity.value = value;
  curlNoiseFlowMaterial.uniforms.forceIntensity.value = value;
});
commonFolder.add(params, "decayRate", 0.9, 0.999).onChange((value) => {
  fluidDynamicsMaterial.uniforms.decayRate.value = value;
  curlNoiseFlowMaterial.uniforms.decayRate.value = value;
});
commonFolder.add(params, "feedbackStrength", 0.1, 0.99).onChange((value) => {
  fluidDynamicsMaterial.uniforms.feedbackStrength.value = value;
  curlNoiseFlowMaterial.uniforms.feedbackStrength.value = value;
});
commonFolder.open();

// 1. Fluid dynamics parameters
const fluidFolder = gui.addFolder("Fluid Dynamics Parameters");
fluidFolder.add(params, "viscosity", 0.5, 0.99).onChange((value) => {
  fluidDynamicsMaterial.uniforms.viscosity.value = value;
});
fluidFolder.add(params, "velocityScale", 0.1, 5.0).onChange((value) => {
  fluidDynamicsMaterial.uniforms.velocityScale.value = value;
});
fluidFolder.add(params, "pressureIterations", 1, 32, 1).onChange((value) => {
  fluidDynamicsMaterial.uniforms.pressureIterations.value = value;
});

// 2. Curl noise flow parameters
const curlFolder = gui.addFolder("Curl Noise Flow Parameters");
curlFolder.add(params, "noiseScale", 0.1, 3.0).onChange((value) => {
  curlNoiseFlowMaterial.uniforms.noiseScale.value = value;
});
curlFolder.add(params, "noiseSpeed", 0.01, 1.0).onChange((value) => {
  curlNoiseFlowMaterial.uniforms.noiseSpeed.value = value;
});
curlFolder.add(params, "noiseOctaves", 1, 8, 1).onChange((value) => {
  curlNoiseFlowMaterial.uniforms.noiseOctaves.value = value;
});
curlFolder.add(params, "noiseTwist", 0.0, 5.0).onChange((value) => {
  curlNoiseFlowMaterial.uniforms.noiseTwist.value = value;
});

// Color settings
const colorFolder = gui.addFolder("Color Settings");
colorFolder.addColor(params, "primaryColor").onChange((value) => {
  renderMaterial.uniforms.primaryColor.value.set(...value);
});
colorFolder.addColor(params, "accentColor").onChange((value) => {
  renderMaterial.uniforms.accentColor.value.set(...value);
});
colorFolder.add(params, "colorIntensity", 1.0, 10.0).onChange((value) => {
  renderMaterial.uniforms.colorIntensity.value = value;
});
colorFolder.add(params, "colorSaturation", 0.0, 2.0).onChange((value) => {
  renderMaterial.uniforms.colorSaturation.value = value;
});
colorFolder.open();

// Auto movement parameters
const movementFolder = gui.addFolder("Auto Movement");
movementFolder.add(params, "autoMove").onChange((value) => {
  fluidDynamicsMaterial.uniforms.autoMove.value = value;
  curlNoiseFlowMaterial.uniforms.autoMove.value = value;
});
movementFolder.add(params, "autoMoveWhenInactive").onChange((value) => {
  fluidDynamicsMaterial.uniforms.autoMoveWhenInactive.value = value;
  curlNoiseFlowMaterial.uniforms.autoMoveWhenInactive.value = value;
});
movementFolder.add(params, "movementRadius", 0.1, 1.0).onChange((value) => {
  fluidDynamicsMaterial.uniforms.movementRadius.value = value;
  curlNoiseFlowMaterial.uniforms.movementRadius.value = value;
});
movementFolder.add(params, "movementSpeed", 0.01, 1.0).onChange((value) => {
  fluidDynamicsMaterial.uniforms.movementSpeed.value = value;
  curlNoiseFlowMaterial.uniforms.movementSpeed.value = value;
});
movementFolder.open();

// Handle window resize
window.addEventListener("resize", () => {
  const width = window.innerWidth;
  const height = window.innerHeight;

  renderer.setSize(width, height);
  renderTarget1.setSize(width, height);
  renderTarget2.setSize(width, height);

  fluidDynamicsMaterial.uniforms.iResolution.value.set(width, height);
  curlNoiseFlowMaterial.uniforms.iResolution.value.set(width, height);
  renderMaterial.uniforms.iResolution.value.set(width, height);
});

// Initialize first frame
let currentTarget = renderTarget1;
let previousTarget = renderTarget2;
let frameCount = 0;

// Update active simulation with the default (fluid dynamics)
updateSimulationType(params.simulationType);

// Animation loop
function animate() {
  requestAnimationFrame(animate);

  // Update time uniform
  const time = performance.now() * 0.001;

  // Update all simulation time uniforms
  fluidDynamicsMaterial.uniforms.iTime.value = time;
  curlNoiseFlowMaterial.uniforms.iTime.value = time;
  renderMaterial.uniforms.iTime.value = time;

  // Check if mouse is inactive for a while
  const inactiveTime = time - lastMouseMoveTime / 1000;
  const isInactive = inactiveTime > 1.0;

  // Update mouse uniform for all simulations
  const mouseUniform = new THREE.Vector4(
    mouse.x,
    mouse.y,
    mouseActive && !isInactive ? 1.0 : 0.0,
    0.0
  );

  // Update mouse uniform in all simulations
  fluidDynamicsMaterial.uniforms.iMouse.value.copy(mouseUniform);
  curlNoiseFlowMaterial.uniforms.iMouse.value.copy(mouseUniform);

  // Update frame counter for active simulation
  simulationQuad.material.uniforms.iFrame.value = frameCount++;

  // Pass previous frame to active simulation
  simulationQuad.material.uniforms.iChannel0.value = previousTarget.texture;

  // Render simulation to current target
  renderer.setRenderTarget(currentTarget);
  renderer.render(simulationScene, camera);

  // Render visualization using the simulation result
  renderMaterial.uniforms.iChannel0.value = currentTarget.texture;
  renderer.setRenderTarget(null);
  renderer.render(renderScene, camera);

  // Swap buffers
  const temp = currentTarget;
  currentTarget = previousTarget;
  previousTarget = temp;

  // Update stats
  stats.update();
}

animate();
