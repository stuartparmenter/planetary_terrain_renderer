import bevy_render::view::View;
import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_pbr::render::view_transformations::uv_to_ndc;

// Mirror of Bevy's `PreviousViewUniforms` (bevy_core_pipeline::prepass).
struct PreviousViewUniforms {
    view_from_world: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    clip_from_view: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> previous_view: PreviousViewUniforms;
@group(0) @binding(2) var terrain_depth: texture_depth_2d;

struct Output {
    // Written as the fragment depth so the pipeline's GreaterEqual test
    // against the final scene depth rejects terrain that a mesh occludes
    // (reversed-Z: a nearer mesh has a larger stored depth). Depth writes
    // are disabled, so this only gates the write, never mutates the depth.
    @builtin(frag_depth) depth: f32,
    @location(0) motion: vec4<f32>,
}

// Writes camera-motion motion vectors for terrain pixels so temporal
// upscalers (DLSS/TAA) reproject the terrain in step with the meshes,
// which write their own motion vectors in Bevy's prepass.
//
// The terrain is static, so its screen-space motion is entirely
// camera-induced. Reconstructing the world position from the terrain's
// *rendered* depth makes this exact and folds in the CDLOD morph for
// free. The depth test (above) restricts writes to terrain-frontmost
// pixels, so mesh / background motion vectors are never clobbered.
// Mirrors Bevy's `background_motion_vectors`, restricted to terrain depth.
@fragment
fn fragment(in: FullscreenVertexOutput) -> Output {
    let coord = vec2<i32>(in.position.xy);
    let td = textureLoad(terrain_depth, coord, 0);

    // No terrain here (cleared depth == 0): leave the existing vector.
    if (td == 0.0) {
        discard;
    }

    let world_pos_h = view.world_from_clip * vec4(uv_to_ndc(in.uv), td, 1.0);
    let world_pos = vec4(world_pos_h.xyz / world_pos_h.w, 1.0);

    // Unjittered on both sides so the TAA/DLSS jitter cancels out.
    let curr = view.unjittered_clip_from_world * world_pos;
    let prev = previous_view.unjittered_clip_from_world * world_pos;

    var out: Output;
    out.depth = td;
    out.motion = vec4((curr.xy / curr.w - prev.xy / prev.w) * vec2(0.5, -0.5), 0.0, 1.0);
    return out;
}
