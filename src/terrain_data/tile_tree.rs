use crate::{
    floating_origin::view_local_position,
    math::{Coordinate, TerrainShape, TileCoordinate},
    render::TerrainViewUniform,
    terrain::TerrainConfig,
    terrain_data::{
        INVALID_ATLAS_INDEX, INVALID_LOD, TerrainTileDropped, TerrainTileReady, TileAtlas,
    },
    terrain_view::{TerrainViewComponents, TerrainViewConfig},
};
use bevy::{
    asset::RenderAssetUsages,
    ecs::entity::EntityHashSet,
    math::{DVec2, DVec3, primitives::ViewFrustum},
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::BufferUsages,
        storage::ShaderBuffer,
    },
};
#[cfg(feature = "big_space")]
use big_space::prelude::{CellCoord, Grids};
use itertools::iproduct;
use ndarray::Array4;
use std::{cmp::Ordering, iter};

/// View movement (in meters) beyond which [`TileTree::update`] must run again.
/// `update` is the only place `view_coordinates` refresh, and they anchor the
/// approximate-height sample (`prepare_prepass.wgsl`); a stale anchor misreads
/// the height by up to the distance moved times the slope, which inflates the
/// near-field view distance and collapses the morph/subdivision target lod. A
/// couple of meters keeps that error below eye height on any plausible slope,
/// while still skipping the recompute for a stationary view.
const VIEW_UPDATE_THRESHOLD: f64 = 2.0;

/// The current state of a tile of a [`TileTree`].
///
/// This indicates, whether or not the tile should be loaded into the [`TileAtlas`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RequestState {
    /// The tile should be loaded.
    Requested,
    /// The tile does not have to be loaded.
    Released,
}

/// The internal representation of a tile in a [`TileTree`].
struct TileState {
    /// The current tile coordinate at the tile_tree position.
    coordinate: TileCoordinate,
    /// Indicates, whether the tile is currently demanded or released.
    state: RequestState,
}

impl Default for TileState {
    fn default() -> Self {
        Self {
            coordinate: TileCoordinate::INVALID,
            state: RequestState::Released,
        }
    }
}

/// An entry of the [`TileTree`], used to access the best currently loaded tile
/// of the [`TileAtlas`] on the CPU.
///
/// These entries are synced each frame with their equivalent representations in the
/// [`GpuTileTree`](super::gpu_tile_tree::GpuTileTree) for access on the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TileTreeEntry {
    /// The atlas index of the best entry.
    pub(crate) atlas_index: u32,
    /// The atlas lod of the best entry.
    pub(crate) atlas_lod: u32,
}

impl Default for TileTreeEntry {
    fn default() -> Self {
        Self {
            atlas_index: INVALID_ATLAS_INDEX,
            atlas_lod: INVALID_LOD,
        }
    }
}

#[derive(Component)]
pub struct TerrainViewKey(pub (Entity, Entity));

/// A quadtree-like view of a terrain, that requests and releases tiles from the [`TileAtlas`]
/// depending on the distance to the viewer.
///
/// It can be used to access the best currently loaded tile of the [`TileAtlas`].
/// Additionally its sends this data to the GPU via the
/// [`GpuTileTree`](super::gpu_tile_tree::GpuTileTree) so that it can be utilised
/// in shaders as well.
///
/// Each view (camera, shadow-casting light) that should consider the terrain has to
/// have an associated tile tree.
///
/// This tile tree is a "cube" with a size of (`tree_size`x`tree_size`x`lod_count`), where each layer
/// corresponds to a lod. These layers are wrapping (modulo `tree_size`), that means that
/// the tile tree is always centered under the viewer and only considers `tree_size` / 2 tiles
/// in each direction.
///
/// Each frame the tile tree determines the state of each tile via the
/// `compute_requests` methode.
/// After the [`TileAtlas`] has adjusted to these requests, the tile tree retrieves the best
/// currently loaded tiles from the tile atlas via the `adjust` methode, which can later be used to access the terrain data.
#[derive(Component)]
pub struct TileTree {
    /// The current cpu tile_tree data. This is synced each frame with the gpu tile_tree data.
    pub(crate) data: Array4<TileTreeEntry>,
    /// Tiles that are no longer required by this tile_tree.
    pub(crate) released_tiles: Vec<TileCoordinate>,
    /// Tiles that are requested to be loaded by this tile_tree.
    pub(crate) requested_tiles: Vec<TileCoordinate>,
    /// The internal tile states of the tile_tree.
    tiles: Array4<TileState>,
    /// The count of tiles in x and y direction per layer.
    pub(crate) tree_size: u32,
    pub(crate) lod_count: u32,
    pub(crate) shape: TerrainShape,
    pub(crate) geometry_tile_count: u32,
    pub(crate) refinement_count: u32,
    pub(crate) grid_size: u32,
    pub(crate) morph_range: f32,
    pub(crate) blend_range: f32,
    pub(crate) morph_distance: f64,
    pub(crate) blend_distance: f64,
    pub(crate) subdivision_distance: f64,
    pub(crate) load_distance: f64,
    pub(crate) precision_distance: f64,
    pub(crate) view_face: u32,
    pub(crate) view_lod: u32,
    pub(crate) view_local_position: DVec3,
    pub(crate) view_world_position: Vec3,
    pub(crate) view_coordinates: [Coordinate; 6],
    pub(crate) half_spaces: [Vec4; 6],
    pub(crate) surface_approximation: [crate::math::SurfaceApproximation; 6],
    pub(crate) approximate_height: f32,
    pub(crate) order: u32,

    /// View position and approximate height the last time [`Self::update`] ran;
    /// while the view stays within [`VIEW_UPDATE_THRESHOLD`] of it, the
    /// tile-state recompute is skipped (see [`Self::refresh_view`]).
    last_update_view: Option<(DVec3, f32)>,
    /// Whether [`Self::update`] recomputed the tile states this frame.
    updated_this_frame: bool,
    /// Whether [`Self::adjust_to_tile_atlas`] rewrote `data` this frame.
    adjusted_this_frame: bool,
    /// Whether any input of the [`TerrainViewUniform`] changed this frame.
    view_changed_this_frame: bool,

    pub(crate) tile_tree_buffer: Handle<ShaderBuffer>,
    pub(crate) terrain_view_buffer: Handle<ShaderBuffer>,
    pub(crate) approximate_height_buffer: Handle<ShaderBuffer>,
}

impl TileTree {
    /// Creates a new tile_tree from a terrain and a terrain view config.
    pub fn new(
        config: &TerrainConfig,
        view_config: &TerrainViewConfig,
        terrain_view: (Entity, Entity),
        commands: &mut Commands,
        buffers: &mut Assets<ShaderBuffer>, // Todo: solve this dependency with a component hook in the future
    ) -> Self {
        let data = Array4::default((
            config.shape.face_count() as usize,
            config.lod_count as usize,
            view_config.tree_size as usize,
            view_config.tree_size as usize,
        ));

        let terrain_view_buffer = buffers.add(ShaderBuffer::with_size(
            size_of::<TerrainViewUniform>() as u64,
            RenderAssetUsages::all(),
        ));
        let tile_tree_buffer = buffers.add(ShaderBuffer::with_size(
            (data.len() * size_of::<TileTreeEntry>()) as u64,
            RenderAssetUsages::all(),
        ));

        let mut approximate_height_buffer =
            ShaderBuffer::new(vec![0.0f32], RenderAssetUsages::default());
        approximate_height_buffer.buffer_usage |= BufferUsages::COPY_SRC;
        let approximate_height_buffer = buffers.add(approximate_height_buffer);

        commands
            .spawn((
                TerrainViewKey(terrain_view),
                Readback::buffer(approximate_height_buffer.clone()),
            ))
            .observe(Self::approximate_height_readback);

        let face_size = config.shape.face_size();

        let subdivision_distance =
            view_config.morph_distance * face_size * (1.0 + view_config.subdivision_tolerance);

        // Each `refine_tiles` pass advances the tile list exactly one quadtree
        // level (`subdivide` emits children at `lod + 1`, which only the next
        // pass processes), and a tile that still subdivides in the final pass
        // is neither finalized nor are its children processed — a hole under
        // the viewer. So the loop in `tiling_prepass` has to cover the deepest
        // lod `should_be_divided` can demand: it subdivides while
        // `view_distance < subdivision_distance / 2^(lod + 1)`, and clamps the
        // view distance to the same `MIN_VIEW_DISTANCE` assumed here (see
        // `refine_tiles.wgsl` — the two constants must not drift), so demand
        // stops at lod `ceil(log2(subdivision_distance / MIN_VIEW_DISTANCE))
        // - 1`. Covering lods `0..=deepest_lod` takes `deepest_lod + 1` passes
        // (pass N processes lod N - 1). The configured count stays as an upper
        // bound, so terrains large enough to exceed it (planetary scale) keep
        // their configured depth.
        const MIN_VIEW_DISTANCE: f64 = 0.1;
        let deepest_lod = ((subdivision_distance / MIN_VIEW_DISTANCE).log2().ceil() - 1.0)
            .max((config.lod_count - 1) as f64) as u32;
        let refinement_count = view_config.refinement_count.min(deepest_lod + 1);

        Self {
            tree_size: view_config.tree_size,
            lod_count: config.lod_count,
            shape: config.shape,
            geometry_tile_count: view_config.geometry_tile_count,
            refinement_count,
            grid_size: view_config.grid_size,
            morph_distance: view_config.morph_distance * face_size,
            blend_distance: view_config.blend_distance * face_size,
            load_distance: view_config.blend_distance
                * face_size
                * (1.0 + view_config.load_tolerance),
            subdivision_distance,
            morph_range: view_config.morph_range,
            blend_range: view_config.blend_range,
            precision_distance: view_config.precision_distance * config.shape.scale_scalar(),
            view_face: 0,
            view_lod: view_config.view_lod,
            view_local_position: default(),
            view_world_position: default(),
            data,
            tiles: Array4::default((
                config.shape.face_count() as usize,
                config.lod_count as usize,
                view_config.tree_size as usize,
                view_config.tree_size as usize,
            )),
            released_tiles: default(),
            requested_tiles: default(),
            view_coordinates: default(),
            half_spaces: default(),

            surface_approximation: default(),
            approximate_height: 0.0,
            order: view_config.order,
            last_update_view: None,
            updated_this_frame: true,
            adjusted_this_frame: true,
            view_changed_this_frame: true,
            tile_tree_buffer,
            terrain_view_buffer,
            approximate_height_buffer,
        }
    }

    fn compute_tree_xy(coordinate: Coordinate, tile_count: f64) -> DVec2 {
        // scale and clamp the coordinate to the tile tree bounds
        (coordinate.uv * tile_count).min(DVec2::splat(tile_count - 0.000001))
    }

    fn compute_origin(&self, view_coordinate: Coordinate, lod: u32) -> IVec2 {
        let tile_count = (lod as f64).exp2();
        let tree_xy = Self::compute_tree_xy(view_coordinate, tile_count);

        (tree_xy - 0.5 * self.tree_size as f64)
            .round()
            .clamp(
                DVec2::splat(0.0),
                DVec2::splat(tile_count - self.tree_size as f64),
            )
            .as_ivec2()
    }

    fn compute_tile_distance(&self, tile: TileCoordinate, view_coordinate: Coordinate) -> f64 {
        let tile_count = (tile.lod as f64).exp2();
        let view_tile_xy = Self::compute_tree_xy(view_coordinate, tile_count);
        let tile_offset = view_tile_xy.as_ivec2() - tile.xy;
        let mut offset = view_tile_xy % 1.0;

        offset.x = match tile_offset.x.cmp(&0) {
            Ordering::Less => 0.0,
            Ordering::Greater => 1.0,
            Ordering::Equal => offset.x,
        };

        offset.y = match tile_offset.y.cmp(&0) {
            Ordering::Less => 0.0,
            Ordering::Greater => 1.0,
            Ordering::Equal => offset.y,
        };

        let tile_local_position =
            Coordinate::new(tile.face, (tile.xy.as_dvec2() + offset) / tile_count)
                .local_position(self.shape, self.approximate_height);

        tile_local_position.distance(self.view_local_position)
    }

    fn update(&mut self) {
        let view_coordinate = Coordinate::from_local_position(self.view_local_position, self.shape);
        self.view_face = view_coordinate.face;

        for face in 0..self.shape.face_count() {
            let view_coordinate = view_coordinate.project_to_face(face);
            self.view_coordinates[face as usize] = view_coordinate;

            for lod in 0..self.lod_count {
                let origin = self.compute_origin(view_coordinate, lod);

                for (x, y) in iproduct!(0..self.tree_size, 0..self.tree_size) {
                    let tile_coordinate = TileCoordinate {
                        face,
                        lod,
                        xy: origin + IVec2::new(x as i32, y as i32),
                    };

                    let tile_distance =
                        self.compute_tile_distance(tile_coordinate, view_coordinate);
                    let load_distance = self.load_distance / (tile_coordinate.lod as f64).exp2();

                    let state = if lod == 0 || tile_distance < load_distance {
                        RequestState::Requested
                    } else {
                        RequestState::Released
                    };

                    let tile = &mut self.tiles[[
                        face as usize,
                        lod as usize,
                        tile_coordinate.xy.x as usize % self.tree_size as usize,
                        tile_coordinate.xy.y as usize % self.tree_size as usize,
                    ]];

                    // check if tile_tree slot refers to a new tile
                    if tile_coordinate != tile.coordinate {
                        // release old tile
                        if tile.state == RequestState::Requested {
                            tile.state = RequestState::Released;
                            self.released_tiles.push(tile.coordinate);
                        }

                        tile.coordinate = tile_coordinate;
                    }

                    // request or release tile based on its distance to the view
                    match (tile.state, state) {
                        (RequestState::Released, RequestState::Requested) => {
                            tile.state = RequestState::Requested;
                            self.requested_tiles.push(tile.coordinate);
                        }
                        (RequestState::Requested, RequestState::Released) => {
                            tile.state = RequestState::Released;
                            self.released_tiles.push(tile.coordinate);
                        }
                        (_, _) => {}
                    }
                }
            }
        }
    }

    /// Refreshes the per-frame view inputs and recomputes the tile states when
    /// the view has strayed farther than [`VIEW_UPDATE_THRESHOLD`] from where
    /// [`Self::update`] last ran.
    ///
    /// Tile requests and releases originate exclusively in `update`, so it is
    /// safe to skip while the view is stationary; `approximate_height` offsets
    /// the surface the tile distances are measured against, so a
    /// height-readback jump counts as movement too.
    fn refresh_view(
        &mut self,
        view_local_position: DVec3,
        view_world_position: Vec3,
        half_spaces: [Vec4; 6],
    ) {
        self.view_changed_this_frame = view_local_position != self.view_local_position
            || view_world_position != self.view_world_position
            || half_spaces != self.half_spaces;

        self.view_local_position = view_local_position;
        self.view_world_position = view_world_position;
        self.half_spaces = half_spaces;

        self.updated_this_frame = self.last_update_view.is_none_or(|(position, height)| {
            position.distance_squared(view_local_position)
                >= VIEW_UPDATE_THRESHOLD * VIEW_UPDATE_THRESHOLD
                || (height - self.approximate_height).abs() as f64 >= VIEW_UPDATE_THRESHOLD
        });

        if self.updated_this_frame {
            self.last_update_view = Some((view_local_position, self.approximate_height));
            self.update();
        }
    }

    /// Traverses all tile_trees and updates the tile states,
    /// while selecting newly requested and released tiles.
    #[cfg(feature = "big_space")]
    pub(crate) fn compute_requests(
        camera: Query<&Camera>,
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        grids: Grids,
        views: Query<(&GlobalTransform, &Transform, &CellCoord)>,
    ) {
        for (&(_, view), tile_tree) in tile_trees.iter_mut() {
            let camera = camera.get(view).unwrap();
            let (global_transform, transform, cell) = views.get(view).unwrap();

            let clip_from_view = camera.clip_from_view();
            let world_from_view = global_transform.to_matrix();
            let clip_from_world = clip_from_view * world_from_view.inverse();

            let half_spaces = ViewFrustum::from_clip_from_world(&clip_from_world)
                .half_spaces
                .map(|space| space.normal_d());

            tile_tree.refresh_view(
                view_local_position(&grids, view, transform, cell),
                transform.translation,
                half_spaces,
            );
        }
    }

    #[cfg(not(feature = "big_space"))]
    pub(crate) fn compute_requests(
        camera: Query<&Camera>,
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        views: Query<(&GlobalTransform, &Transform)>,
    ) {
        for (&(_, view), tile_tree) in tile_trees.iter_mut() {
            let camera = camera.get(view).unwrap();
            let (global_transform, transform) = views.get(view).unwrap();

            let clip_from_view = camera.clip_from_view();
            let world_from_view = global_transform.to_matrix();
            let clip_from_world = clip_from_view * world_from_view.inverse();

            let half_spaces = ViewFrustum::from_clip_from_world(&clip_from_world)
                .half_spaces
                .map(|space| space.normal_d());

            tile_tree.refresh_view(
                view_local_position(global_transform),
                transform.translation,
                half_spaces,
            );
        }
    }

    /// Adjusts all tile_trees to their corresponding tile atlas
    /// by updating the entries with the best available tiles.
    pub(crate) fn adjust_to_tile_atlas(
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        tile_atlases: Query<&TileAtlas>,
        mut ready_messages: MessageReader<TerrainTileReady>,
        mut dropped_messages: MessageReader<TerrainTileDropped>,
    ) {
        // For an unchanged tile grid, `get_best_tile` output only changes at
        // residency edges — a tile flipping to `Loaded` or a loaded slot being
        // evicted — which are exactly what the ready/dropped messages report.
        // Trees whose `update` was skipped therefore only need re-adjusting
        // when their terrain reported such an edge this frame.
        let changed_terrains: EntityHashSet = ready_messages
            .read()
            .map(|message| message.terrain)
            .chain(dropped_messages.read().map(|message| message.terrain))
            .collect();

        for (&(terrain, _view), tile_tree) in tile_trees.iter_mut() {
            tile_tree.adjusted_this_frame =
                tile_tree.updated_this_frame || changed_terrains.contains(&terrain);
            if !tile_tree.adjusted_this_frame {
                continue;
            }

            let tile_atlas = tile_atlases.get(terrain).unwrap();

            for (tile, entry) in iter::zip(&tile_tree.tiles, &mut tile_tree.data) {
                *entry = tile_atlas.get_best_tile(tile.coordinate);
            }
        }
    }

    pub fn generate_surface_approximation(mut tile_trees: ResMut<TerrainViewComponents<TileTree>>) {
        for tile_tree in tile_trees.values_mut() {
            tile_tree.surface_approximation = tile_tree.view_coordinates.map(|view_coordinate| {
                crate::math::SurfaceApproximation::compute(
                    view_coordinate,
                    tile_tree.view_local_position,
                    tile_tree.view_world_position,
                    tile_tree.shape,
                )
            });
        }
    }

    pub fn update_terrain_view_buffer(
        tile_trees: Res<TerrainViewComponents<TileTree>>,
        mut buffers: ResMut<Assets<ShaderBuffer>>,
    ) {
        for tile_tree in tile_trees.values() {
            // Nothing feeding the two buffers changed this frame: the uniform's
            // inputs are bit-identical and `data` was left untouched by both
            // `update` and `adjust_to_tile_atlas`. Skipping the `get_mut` also
            // avoids marking the assets modified, which alone would re-upload
            // both GPU buffers; an unmodified asset keeps its previously
            // prepared GPU buffer, so views stay valid.
            // `adjusted_this_frame` is set to `updated_this_frame || …`, so it
            // already implies `updated_this_frame` — testing both is redundant.
            if !tile_tree.adjusted_this_frame && !tile_tree.view_changed_this_frame {
                continue;
            }

            {
                let mut terrain_view_buffer =
                    buffers.get_mut(&tile_tree.terrain_view_buffer).unwrap();
                terrain_view_buffer.clear();
                terrain_view_buffer.extend_from_slice(&[TerrainViewUniform::from(tile_tree)]);
            }
            {
                let mut tile_tree_buffer = buffers.get_mut(&tile_tree.tile_tree_buffer).unwrap();
                tile_tree_buffer.clear();
                tile_tree_buffer.extend(tile_tree.data.clone().into_iter());
            }
        }
    }

    pub fn approximate_height_readback(
        on: On<ReadbackComplete>,
        terrain_view: Query<&TerrainViewKey>,
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
    ) {
        let entity = on.event().entity;
        // A readback queued before the terrain despawned can still complete
        // after its tile tree is gone.
        let Ok(TerrainViewKey(terrain_view)) = terrain_view.get(entity) else {
            return;
        };
        let Some(tile_tree) = tile_trees.get_mut(terrain_view) else {
            return;
        };
        tile_tree.approximate_height = on.event().to_shader_type();
    }
}
