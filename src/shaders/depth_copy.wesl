import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;

// The terrain depth texture is multisampled iff the view is. The pipeline
// (see `terrain_pass.rs::DepthCopyPipeline`) sets `MULTISAMPLED`
// accordingly; `textureLoad`'s third argument is the sample index for a
// multisampled texture and the mip level otherwise, and 0 is correct for
// both.
@if(MULTISAMPLED)
@group(0) @binding(0)
var depth_texture: texture_depth_multisampled_2d;
@else
@group(0) @binding(0)
var depth_texture: texture_depth_2d;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @builtin(frag_depth) f32 {
    return textureLoad(depth_texture, vec2<u32>(in.uv * vec2<f32>(textureDimensions(depth_texture))), 0);
}
