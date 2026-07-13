#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

// Restages the terrain's privately rendered G-buffer into Bevy's deferred
// prepass attachments. The terrain draw can't target those attachments
// directly: on deferred views the mesh-view bind group (group 0 of the
// terrain draw) itself contains the current-frame deferred texture, and a
// texture can't be both bound and attached in the same pass. The pipeline's
// depth test (`Greater`, reverse-Z) gates every output — where the terrain
// isn't the closest surface written so far, neither the G-buffer texels nor
// the depth land.
@group(0) @binding(0)
var terrain_gbuffer: texture_2d<u32>;
@group(0) @binding(1)
var terrain_lighting_pass_id: texture_2d<u32>;
@group(0) @binding(2)
var terrain_depth: texture_depth_2d;

struct FragmentOutput {
    @location(0) deferred: vec4<u32>,
    @location(1) deferred_lighting_pass_id: u32,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> FragmentOutput {
    let pixel = vec2<u32>(in.position.xy);
    return FragmentOutput(
        textureLoad(terrain_gbuffer, pixel, 0),
        textureLoad(terrain_lighting_pass_id, pixel, 0).x,
        textureLoad(terrain_depth, pixel, 0),
    );
}
