struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = input.position;
    output.color = input.color;
    output.uv = input.uv;
    return output;
}

@group(2) @binding(0)
var psp_texture: texture_2d<f32>;

@group(2) @binding(1)
var psp_sampler: sampler;

struct FragmentUniforms {
    function: u32,
    flags: u32,
    alpha_test: u32,
    alpha_function: u32,
    alpha_reference: u32,
    alpha_mask: u32,
    color_test: u32,
    color_function: u32,
    color_reference: u32,
    color_mask: u32,
    _padding: vec2<u32>,
    environment: vec4<f32>,
};

@group(3) @binding(0)
var<uniform> uniforms: FragmentUniforms;

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texture_color = textureSample(psp_texture, psp_sampler, input.uv);
    let vertex_rgb = input.color.rgb;
    let texture_rgb = texture_color.rgb;
    let doubled = (uniforms.flags & 2u) != 0u;
    let scale = select(1.0, 2.0, doubled);
    var rgb = vertex_rgb * texture_rgb * scale;
    switch uniforms.function {
        case 1u: {
            rgb = select(texture_rgb, mix(vertex_rgb, texture_rgb, texture_color.a),
                (uniforms.flags & 1u) != 0u) * scale;
        }
        case 2u: {
            rgb = mix(vertex_rgb, uniforms.environment.rgb, texture_rgb) * scale;
        }
        case 3u: {
            rgb = texture_rgb * scale;
        }
        default: {
            if (uniforms.function >= 4u) {
                rgb = (vertex_rgb + texture_rgb) * scale;
            }
        }
    }
    let output_rgb = min(rgb, vec3<f32>(1.0));
    if (uniforms.color_test != 0u) {
        let color_value = u32(round(output_rgb.r * 255.0))
            | (u32(round(output_rgb.g * 255.0)) << 8u)
            | (u32(round(output_rgb.b * 255.0)) << 16u);
        let value = color_value & uniforms.color_mask;
        let reference = uniforms.color_reference & uniforms.color_mask;
        var passed = false;
        switch uniforms.color_function {
            case 0u: { passed = false; }
            case 1u: { passed = true; }
            case 2u: { passed = value == reference; }
            default: { passed = value != reference; }
        }
        if (!passed) {
            discard;
        }
    }
    // alpha testing uses the same combined alpha that blending receives.
    let alpha = select(input.color.a, input.color.a * texture_color.a, (uniforms.flags & 1u) != 0u);
    if (uniforms.alpha_test != 0u) {
        let alpha_byte = u32(round(clamp(alpha, 0.0, 1.0) * 255.0));
        let value = alpha_byte & uniforms.alpha_mask;
        let reference = uniforms.alpha_reference & uniforms.alpha_mask;
        var passed = false;
        switch uniforms.alpha_function {
            case 0u: { passed = false; }
            case 1u: { passed = true; }
            case 2u: { passed = value == reference; }
            case 3u: { passed = value != reference; }
            case 4u: { passed = value < reference; }
            case 5u: { passed = value <= reference; }
            case 6u: { passed = value > reference; }
            default: { passed = value >= reference; }
        }
        if (!passed) {
            discard;
        }
    }
    return vec4<f32>(output_rgb, alpha);
}
