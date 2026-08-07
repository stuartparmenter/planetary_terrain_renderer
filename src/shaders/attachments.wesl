import bevy_terrain::shaders::types::{AtlasTile, TangentSpace, AttachmentConfig, SampleUV, WorldCoordinate};
import bevy_terrain::shaders::bindings::{terrain_data, terrain_view, terrain_sampler, height_attachment};

@if(FRAGMENT)
fn compute_sample_uv(tile: AtlasTile, attachment: AttachmentConfig) -> SampleUV {
    let uv    = tile.coordinate.uv * attachment.scale + attachment.offset;
    let lod   = log2(attachment.texture_size * max(length(tile.coordinate.uv_dx), length(tile.coordinate.uv_dy)));
    let scale = exp2(max(1.5 * tile.blend_ratio - lod, 0.0));
    let dx    = tile.coordinate.uv_dx * scale;
    let dy    = tile.coordinate.uv_dy * scale;

    return SampleUV(uv, dx, dy);
}
@if(!FRAGMENT)
fn compute_sample_uv(tile: AtlasTile, attachment: AttachmentConfig) -> SampleUV {
    let uv = tile.coordinate.uv * attachment.scale + attachment.offset;

    return SampleUV(uv);
}

// Samples an arbitrary attachment at the tile's coordinate, applying the same LOD / blend
// handling as height sampling. Pass the attachment's config (`terrain_data.attachments.<label>`)
// together with its texture global (`<label>_attachment`), e.g.
//
//     let density = sample_attachment(terrain_data.attachments.grass_density, grass_density_attachment, tile).x;
//
// This uses linear filtering, so it is meant for continuous data (density, blend weights, ...).
// For integer id attachments, gather the raw texels instead (see `compute_sample_uv` +
// `textureGather` / `textureLoad`), since filtering would blend distinct ids into meaningless
// intermediate values.
fn sample_attachment(attachment: AttachmentConfig, attachment_texture: texture_2d_array<f32>, tile: AtlasTile) -> vec4<f32> {
    let uv = compute_sample_uv(tile, attachment);

    @if(FRAGMENT && SAMPLE_GRAD) {
        return textureSampleGrad(attachment_texture, terrain_sampler, uv.uv, tile.index, uv.dx, uv.dy);
    } @elif(FRAGMENT) {
        return textureSampleLevel(attachment_texture, terrain_sampler, uv.uv, tile.index, tile.blend_ratio);
    } @else {
        return textureSampleLevel(attachment_texture, terrain_sampler, uv.uv, tile.index, 0.0);
    }
}

fn sample_height(tile: AtlasTile) -> f32 {
    return terrain_data.terrain.height_scale * sample_attachment(terrain_data.attachments.height, height_attachment, tile).x;
}

fn sample_height_mask(tile: AtlasTile) -> bool {
    let attachment = terrain_data.attachments.height;

    if (attachment.mask == 0) { return false; }

    let uv         = tile.coordinate.uv * attachment.scale + attachment.offset;
    let raw_height = textureGather(0, height_attachment, terrain_sampler, uv, tile.index);
    let mask       = bitcast<vec4<u32>>(raw_height) & vec4<u32>(1);

    return any(mask == vec4<u32>(0));
}

@if(FRAGMENT)
fn sample_surface_gradient(tile: AtlasTile, tangent_space: TangentSpace) -> vec3<f32> {
    let attachment = terrain_data.attachments.height;
    let uv         = compute_sample_uv(tile, attachment);
    let scale      = max(length(uv.dx), length(uv.dy));
    let step       = 0.5 * scale;

    @if(SAMPLE_GRAD)
    let height   = textureSampleGrad(height_attachment, terrain_sampler, uv.uv + vec2<f32>(-step, -step), tile.index, uv.dx, uv.dy).x;
    @if(SAMPLE_GRAD)
    let height_u = textureSampleGrad(height_attachment, terrain_sampler, uv.uv + vec2<f32>( step, -step), tile.index, uv.dx, uv.dy).x;
    @if(SAMPLE_GRAD)
    let height_v = textureSampleGrad(height_attachment, terrain_sampler, uv.uv + vec2<f32>(-step,  step), tile.index, uv.dx, uv.dy).x;
    @if(!SAMPLE_GRAD)
    let height   = textureSampleLevel(height_attachment, terrain_sampler, uv.uv + vec2<f32>(-step, -step), tile.index, tile.blend_ratio).x;
    @if(!SAMPLE_GRAD)
    let height_u = textureSampleLevel(height_attachment, terrain_sampler, uv.uv + vec2<f32>( step, -step), tile.index, tile.blend_ratio).x;
    @if(!SAMPLE_GRAD)
    let height_v = textureSampleLevel(height_attachment, terrain_sampler, uv.uv + vec2<f32>(-step,  step), tile.index, tile.blend_ratio).x;

    var height_duv = vec2<f32>(height_u - height, height_v - height) / scale;

    let start = 0.5;
    let end   = 0.05;
    let lod   = max(0.0, log2(attachment.texture_size * scale));
    let ratio = saturate((lod - start) / (end - start));

    if (ratio > 0.0 && tile.coordinate.lod == terrain_data.terrain.lod_count - 1) {
        let coord       = attachment.texture_size * uv.uv - 0.5;
        let coord_floor = floor(coord);
        let center_uv   = (coord_floor + 0.5) / attachment.texture_size;

        let height_TL = textureGather(0, height_attachment, terrain_sampler, center_uv, tile.index, vec2(-1, -1));
        let height_TR = textureGather(0, height_attachment, terrain_sampler, center_uv, tile.index, vec2( 1, -1));
        let height_BL = textureGather(0, height_attachment, terrain_sampler, center_uv, tile.index, vec2(-1,  1));
        let height_BR = textureGather(0, height_attachment, terrain_sampler, center_uv, tile.index, vec2( 1,  1));
        let height_matrix = mat4x4<f32>(height_TL.w, height_TL.z, height_TR.w, height_TR.z,
                                        height_TL.x, height_TL.y, height_TR.x, height_TR.y,
                                        height_BL.w, height_BL.z, height_BR.w, height_BR.z,
                                        height_BL.x, height_BL.y, height_BR.x, height_BR.y);

        let t  = saturate(coord - coord_floor);
        let A  = vec2<f32>(1.0 - t.x, t.x);
        let B  = vec2<f32>(1.0 - t.y, t.y);
        let X  = 0.25 * vec4<f32>(A.x, 2 * A.x + A.y, A.x + 2 * A.y, A.y);
        let Y  = 0.25 * vec4<f32>(B.x, 2 * B.x + B.y, B.x + 2 * B.y, B.y);
        let dX = 0.5 * vec4<f32>(-A.x, -A.y, A.x, A.y);
        let dY = 0.5 * vec4<f32>(-B.x, -B.y, B.x, B.y);

        let upscaled_height_duv = attachment.texture_size * vec2(dot(Y, dX * height_matrix), dot(dY, X * height_matrix));
        height_duv = mix(height_duv, upscaled_height_duv, ratio);
    }

    let height_dx = dot(height_duv, tile.coordinate.uv_dx);
    let height_dy = dot(height_duv, tile.coordinate.uv_dy);

//    let height_dx = dpdx(height);
//    let height_dy = dpdy(height);

    return terrain_data.terrain.height_scale * tangent_space.scale * (height_dx * tangent_space.tangent_x + height_dy * tangent_space.tangent_y);
}

fn compute_slope(world_normal: vec3<f32>, surface_gradient: vec3<f32>) -> f32 {
    let normal  = normalize(world_normal - surface_gradient);
    let cos_slope = min(dot(normal, world_normal), 1.0); // avoid artifacts
    return acos(cos_slope); // slope in radians
}

fn random_unit_vector(seed: f32) -> vec3<f32> {
    let angle1 = fract(sin(seed * 12.9898) * 43758.5453) * 6.28318;
    let angle2 = fract(cos(seed * 78.233) * 43758.5453) * 3.14159;

    let x = sin(angle2) * cos(angle1);
    let y = sin(angle2) * sin(angle1);
    let z = cos(angle2);

    return vec3<f32>(x, y, z);
}

// Todo: clean up relief shading
fn relief_shading(world_coordinate: WorldCoordinate, surface_gradient: vec3<f32>) -> f32 {
    let seed = 6.0;
    let num_lights = 4;  // Number of lights in the cone
    let theta_max = 0.8; // Max cone angle (radians), adjust for stronger effect
    let scale = 0.5 * log2(world_coordinate.view_distance);
    let normal = normalize(world_coordinate.normal - scale * surface_gradient);

    var total_intensity = 0.0;

    for (var i = 0; i < num_lights; i = i + 1) {
        // Generate a random point in the cone around world_normal
        let rand_offset = random_unit_vector(seed + f32(i));
        let light_dir = normalize(world_coordinate.normal + theta_max * rand_offset);
        total_intensity += max(dot(normal, light_dir), 0.0);
    }

    return total_intensity / f32(num_lights);
}
