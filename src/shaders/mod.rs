use crate::terrain_data::AttachmentLabel;
use bevy::{asset::embedded_asset, prelude::*};
use itertools::Itertools;

pub const DEFAULT_VERTEX_SHADER: &str = "embedded://bevy_terrain/shaders/render/vertex.wesl";
pub const DEFAULT_FRAGMENT_SHADER: &str = "embedded://bevy_terrain/shaders/render/fragment.wesl";
pub const PREPARE_PREPASS_SHADER: &str =
    "embedded://bevy_terrain/shaders/tiling_prepass/prepare_prepass.wesl";
pub const REFINE_TILES_SHADER: &str =
    "embedded://bevy_terrain/shaders/tiling_prepass/refine_tiles.wesl";
// pub(crate) const SPLIT_SHADER: &str = "embedded://bevy_terrain/shaders/preprocess/split.wesl";
// pub(crate) const STITCH_SHADER: &str = "embedded://bevy_terrain/shaders/preprocess/stitch.wesl";
// pub(crate) const DOWNSAMPLE_SHADER: &str =
//     "embedded://bevy_terrain/shaders/preprocess/downsample.wesl";
pub(crate) const PICKING_SHADER: &str = "embedded://bevy_terrain/shaders/picking.wesl";
pub(crate) const DEPTH_COPY_SHADER: &str = "embedded://bevy_terrain/shaders/depth_copy.wesl";
pub(crate) const DEFERRED_COMPOSITE_SHADER: &str =
    "embedded://bevy_terrain/shaders/deferred_composite.wesl";
pub(crate) const TERRAIN_MOTION_SHADER: &str =
    "embedded://bevy_terrain/shaders/terrain_motion.wesl";
pub(crate) const MIP_SHADER: &str = "embedded://bevy_terrain/shaders/mipmap.wesl";
pub(crate) const SHADOW_MAP_SHADER: &str = "embedded://bevy_terrain/shaders/shadow_map.wesl";

#[derive(Default, Resource)]
pub(crate) struct InternalShaders(Vec<Handle<Shader>>);

impl InternalShaders {
    pub(crate) fn load(app: &mut App, shaders: &[&'static str]) {
        let mut shaders = shaders
            .iter()
            .map(|&shader| app.world_mut().resource_mut::<AssetServer>().load(shader))
            .collect_vec();

        let mut internal_shaders = app.world_mut().resource_mut::<InternalShaders>();
        internal_shaders.0.append(&mut shaders);
    }
}

// Todo: this could be implemented using shader defs with values
fn load_bindings_shader(app: &mut App, attachments: &[AttachmentLabel]) {
    let source = include_str!("bindings.wesl");

    let source = (0..8).fold(source.to_string(), |src, i| {
        src.replacen(
            &format!("{{{i}}}"),
            &String::from(
                &attachments
                    .get(i)
                    .cloned()
                    .unwrap_or(AttachmentLabel::Empty(i)),
            ),
            2,
        )
    });

    let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
    // The embedded-style path names the module `bevy_terrain::shaders::bindings`
    // (`from_wesl` derives the import path from it), matching the other
    // embedded terrain shaders even though this one is authored in code.
    let shader = shaders.add(Shader::from_wesl(
        source,
        "embedded://bevy_terrain/shaders/bindings.wesl",
    ));

    let mut internal_shaders = app.world_mut().resource_mut::<InternalShaders>();
    internal_shaders.0.push(shader);
}

pub(crate) fn load_terrain_shaders(app: &mut App, attachments: &[AttachmentLabel]) {
    embedded_asset!(app, "types.wesl");
    embedded_asset!(app, "attachments.wesl");
    embedded_asset!(app, "functions.wesl");
    embedded_asset!(app, "heightfield.wesl");
    embedded_asset!(app, "shadow_map.wesl");
    embedded_asset!(app, "terrain_shadow.wesl");
    embedded_asset!(app, "debug.wesl");
    embedded_asset!(app, "render/vertex.wesl");
    embedded_asset!(app, "render/fragment.wesl");
    embedded_asset!(app, "tiling_prepass/prepare_prepass.wesl");
    embedded_asset!(app, "tiling_prepass/refine_tiles.wesl");
    embedded_asset!(app, "picking.wesl");
    embedded_asset!(app, "depth_copy.wesl");
    embedded_asset!(app, "deferred_composite.wesl");
    embedded_asset!(app, "terrain_motion.wesl");
    embedded_asset!(app, "mipmap.wesl");

    load_bindings_shader(app, attachments);

    InternalShaders::load(
        app,
        &[
            "embedded://bevy_terrain/shaders/types.wesl",
            "embedded://bevy_terrain/shaders/attachments.wesl",
            "embedded://bevy_terrain/shaders/functions.wesl",
            "embedded://bevy_terrain/shaders/heightfield.wesl",
            "embedded://bevy_terrain/shaders/terrain_shadow.wesl",
            "embedded://bevy_terrain/shaders/debug.wesl",
            "embedded://bevy_terrain/shaders/render/vertex.wesl",
            "embedded://bevy_terrain/shaders/render/fragment.wesl",
        ],
    );
}

// pub(crate) fn load_preprocess_shaders(app: &mut App) {
//     embedded_asset!(app, "preprocess/preprocessing.wesl");
//     embedded_asset!(app, "preprocess/split.wesl");
//     embedded_asset!(app, "preprocess/stitch.wesl");
//     embedded_asset!(app, "preprocess/downsample.wesl");
//
//     InternalShaders::load(
//         app,
//         &["embedded://bevy_terrain/shaders/preprocess/preprocessing.wesl"],
//     );
// }
