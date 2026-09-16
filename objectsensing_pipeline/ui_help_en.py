"""English explanations for interactive scientific controls."""

PARAMETER_HELP = {'neighbor_radius': {'summary': 'Local radius for normal and shape analysis.',
                     'detail': 'The effective value is recorded with the run. Compare identical '
                               'inputs using a pinned baseline when changing this setting.',
                     'higher': 'More stable estimates, but nearby edges and thin structures can '
                               'merge.',
                     'lower': 'Finer detail with greater sensitivity to noise and sparse '
                              'neighborhoods.',
                     'cost': 'Inspect stage runtime and memory together with geometric quality; '
                             'the effect depends on scene density and visibility.'},
 'curvature_threshold': {'summary': 'Minimum PCA surface variation before Harris evaluation.',
                         'detail': 'The effective value is recorded with the run. Compare '
                                   'identical inputs using a pinned baseline when changing this '
                                   'setting.',
                         'higher': 'Fewer planar/noisy candidates, but weak rounded corners may be '
                                   'lost.',
                         'lower': 'Greater recall and more wall noise reaching later stages.',
                         'cost': 'Inspect stage runtime and memory together with geometric '
                                 'quality; the effect depends on scene density and visibility.'},
 'harris_k': {'summary': 'Weight of the squared-trace penalty in the Harris response.',
              'detail': 'The effective value is recorded with the run. Compare identical inputs '
                        'using a pinned baseline when changing this setting.',
              'higher': 'Stronger penalty; fewer 3D corners and more candidates in the 2D branch.',
              'lower': 'More permissive detection, including noisy surface responses.',
              'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                      'effect depends on scene density and visibility.'},
 'harris_threshold': {'summary': 'Minimum response for direct 3D corner acceptance.',
                      'detail': 'The effective value is recorded with the run. Compare identical '
                                'inputs using a pinned baseline when changing this setting.',
                      'higher': 'Stronger corners only; lower recall.',
                      'lower': 'More candidates, including noise.',
                      'cost': 'Inspect stage runtime and memory together with geometric quality; '
                              'the effect depends on scene density and visibility.'},
 'harris_reference_neighbors': {'summary': 'Reference neighbor count used to normalize Harris '
                                           'response amplitude.',
                                'detail': 'The effective value is recorded with the run. Compare '
                                          'identical inputs using a pinned baseline when changing '
                                          'this setting.',
                                'higher': 'Changes the response scale and increases separation of '
                                          'volumetric and planar neighborhoods.',
                                'lower': 'Compresses responses; fewer may exceed the current '
                                         'threshold.',
                                'cost': 'Inspect stage runtime and memory together with geometric '
                                        'quality; the effect depends on scene density and '
                                        'visibility.'},
 'convex_hull_ratio': {'summary': 'Threshold for incomplete tangent-plane neighborhoods in the 2D '
                                  'corner test.',
                       'detail': 'The effective value is recorded with the run. Compare identical '
                                 'inputs using a pinned baseline when changing this setting.',
                       'higher': 'Recovers more weak-Harris points, including some partial planar '
                                 'interiors.',
                       'lower': 'Requires stronger boundary evidence and may miss furniture edges.',
                       'cost': 'Inspect stage runtime and memory together with geometric quality; '
                               'the effect depends on scene density and visibility.'},
 'corner_plane_radius_factor': {'summary': 'Multiplier for the wider plane support of a 2D corner.',
                                'detail': 'The effective value is recorded with the run. Compare '
                                          'identical inputs using a pinned baseline when changing '
                                          'this setting.',
                                'higher': 'More robust planar evidence, with a risk of crossing '
                                          'neighboring objects.',
                                'lower': 'More local support; valid edges may lack enough observed '
                                         'area.',
                                'cost': 'Inspect stage runtime and memory together with geometric '
                                        'quality; the effect depends on scene density and '
                                        'visibility.'},
 'corner_plane_min_area': {'summary': 'Minimum planar area supporting a recovered 2D corner.',
                           'detail': 'The effective value is recorded with the run. Compare '
                                     'identical inputs using a pinned baseline when changing this '
                                     'setting.',
                           'higher': 'Rejects small fragments and may remove thin or poorly '
                                     'observed parts.',
                           'lower': 'Recovers small structures with more false support.',
                           'cost': 'Inspect stage runtime and memory together with geometric '
                                   'quality; the effect depends on scene density and visibility.'},
 'corner_plane_normal_cos': {'summary': 'Minimum normal agreement for 2D plane support.',
                             'detail': 'The effective value is recorded with the run. Compare '
                                       'identical inputs using a pinned baseline when changing '
                                       'this setting.',
                             'higher': 'Stricter planes; more sensitive to noisy normals or '
                                       'curvature.',
                             'lower': 'Tolerates noise but may combine different faces.',
                             'cost': 'Inspect stage runtime and memory together with geometric '
                                     'quality; the effect depends on scene density and '
                                     'visibility.'},
 'nms_radius': {'summary': 'Local maximum suppression radius before adjustment.',
                'detail': 'The effective value is recorded with the run. Compare identical inputs '
                          'using a pinned baseline when changing this setting.',
                'higher': 'Fewer, better-spaced keypoints; nearby details can merge.',
                'lower': 'More nearby details and redundant matching work.',
                'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                        'effect depends on scene density and visibility.'},
 'adjust_iterations': {'summary': 'Iterations moving a seed toward a stable geometric '
                                  'intersection.',
                       'detail': 'The effective value is recorded with the run. Compare identical '
                                 'inputs using a pinned baseline when changing this setting.',
                       'higher': 'Better convergence from imprecise seeds, with more cost and '
                                 'drift risk.',
                       'lower': 'Faster and smaller displacements; zero disables adjustment.',
                       'cost': 'Inspect stage runtime and memory together with geometric quality; '
                               'the effect depends on scene density and visibility.'},
 'jitter_reject': {'summary': 'Maximum seed-to-adjusted displacement in meters. Null uses '
                              'automatic behavior.',
                   'detail': 'The effective value is recorded with the run. Compare identical '
                             'inputs using a pinned baseline when changing this setting.',
                   'higher': 'Allows larger corrections but may accept a jump to another '
                             'structure.',
                   'lower': 'Requires local stability and may reject coarse-scan corners.',
                   'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                           'effect depends on scene density and visibility.'},
 'dedup_radius': {'summary': 'Radius merging adjusted keypoints that converge to the same '
                             'location.',
                  'detail': 'The effective value is recorded with the run. Compare identical '
                            'inputs using a pinned baseline when changing this setting.',
                  'higher': 'Removes more duplicates and may merge distinct small details.',
                  'lower': 'Preserves close details and more redundancy.',
                  'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                          'effect depends on scene density and visibility.'},
 'wall_vertical_normal_cos': {'summary': 'Maximum up-axis component for candidate wall normals.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'Tolerates tilted walls and noisy normals, including more '
                                        'furniture faces.',
                              'lower': 'Requires nearly vertical walls and accurate gravity.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'wall_plane_normal_cos': {'summary': 'Normal consistency required within a wall plane.',
                           'detail': 'The effective value is recorded with the run. Compare '
                                     'identical inputs using a pinned baseline when changing this '
                                     'setting.',
                           'higher': 'Cleaner planes but less complete noisy walls.',
                           'lower': 'Denser support with greater risk of merging surfaces.',
                           'cost': 'Inspect stage runtime and memory together with geometric '
                                   'quality; the effect depends on scene density and visibility.'},
 'wall_plane_angle_degrees': {'summary': 'Angular bin width for horizontal wall-normal voting.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'More robust broad bins, but close orientations may merge.',
                              'lower': 'Finer orientations with fragmented noisy support.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'wall_plane_distance': {'summary': 'Metric point-to-plane tolerance for wall hypotheses.',
                         'detail': 'The effective value is recorded with the run. Compare '
                                   'identical inputs using a pinned baseline when changing this '
                                   'setting.',
                         'higher': 'Recovers thick/noisy walls but may absorb adjacent objects.',
                         'lower': 'Sharper localization with more fragile support.',
                         'cost': 'Inspect stage runtime and memory together with geometric '
                                 'quality; the effect depends on scene density and visibility.'},
 'wall_plane_min_points': {'summary': 'Minimum consistent points after wall refinement.',
                           'detail': 'The effective value is recorded with the run. Compare '
                                     'identical inputs using a pinned baseline when changing this '
                                     'setting.',
                           'higher': 'Rejects fragments and may miss undersampled walls.',
                           'lower': 'Accepts sparse support and more false planes.',
                           'cost': 'Inspect stage runtime and memory together with geometric '
                                   'quality; the effect depends on scene density and visibility.'},
 'wall_plane_min_width': {'summary': 'Minimum horizontal wall extent.',
                          'detail': 'The effective value is recorded with the run. Compare '
                                    'identical inputs using a pinned baseline when changing this '
                                    'setting.',
                          'higher': 'Excludes furniture faces but may miss narrow wall sections.',
                          'lower': 'Accepts narrow walls and more object faces.',
                          'cost': 'Inspect stage runtime and memory together with geometric '
                                  'quality; the effect depends on scene density and visibility.'},
 'wall_plane_min_height': {'summary': 'Minimum vertical wall extent.',
                           'detail': 'The effective value is recorded with the run. Compare '
                                     'identical inputs using a pinned baseline when changing this '
                                     'setting.',
                           'higher': 'Fewer furniture faces treated as walls; partial walls may be '
                                     'missed.',
                           'lower': 'More incomplete walls, with a risk of classifying cabinets as '
                                    'walls.',
                           'cost': 'Inspect stage runtime and memory together with geometric '
                                   'quality; the effect depends on scene density and visibility.'},
 'wall_plane_min_area': {'summary': 'Minimum rectangular area of vertical support.',
                         'detail': 'The effective value is recorded with the run. Compare '
                                   'identical inputs using a pinned baseline when changing this '
                                   'setting.',
                         'higher': 'Keeps dominant walls only.',
                         'lower': 'Includes secondary planes and furniture fronts.',
                         'cost': 'Inspect stage runtime and memory together with geometric '
                                 'quality; the effect depends on scene density and visibility.'},
 'wall_plane_max_count': {'summary': 'Maximum dominant vertical planes analyzed.',
                          'detail': 'The effective value is recorded with the run. Compare '
                                    'identical inputs using a pinned baseline when changing this '
                                    'setting.',
                          'higher': 'Covers complex rooms with more comparisons and possible false '
                                    'planes.',
                          'lower': 'Less work; less visible walls may be ignored.',
                          'cost': 'Inspect stage runtime and memory together with geometric '
                                  'quality; the effect depends on scene density and visibility.'},
 'wall_plane_sample_points': {'summary': 'Point budget for initial wall voting.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'More stable votes and greater initial cost.',
                              'lower': 'Faster, with a risk of missing weakly represented walls.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'wall_small_radius': {'summary': 'Radius for immediate planar support around a keypoint.',
                       'detail': 'The effective value is recorded with the run. Compare identical '
                                 'inputs using a pinned baseline when changing this setting.',
                       'higher': 'More stable wall evidence, but nearby object surfaces can mix '
                                 'in.',
                       'lower': 'More local detail and more sensitivity to holes/noise.',
                       'cost': 'Inspect stage runtime and memory together with geometric quality; '
                               'the effect depends on scene density and visibility.'},
 'wall_large_radius': {'summary': 'Radius confirming wall support at a wider scale.',
                       'detail': 'The effective value is recorded with the run. Compare identical '
                                 'inputs using a pinned baseline when changing this setting.',
                       'higher': 'Broader context with more mixing in clutter.',
                       'lower': 'Less cross-object mixing but weaker multi-scale evidence.',
                       'cost': 'Inspect stage runtime and memory together with geometric quality; '
                               'the effect depends on scene density and visibility.'},
 'wall_protrusion_radius': {'summary': 'Radius searching for geometry protruding in front of a '
                                       'wall.',
                            'detail': 'The effective value is recorded with the run. Compare '
                                      'identical inputs using a pinned baseline when changing this '
                                      'setting.',
                            'higher': 'Finds broader objects but may associate unrelated geometry.',
                            'lower': 'Very local protection, weaker on partial furniture.',
                            'cost': 'Inspect stage runtime and memory together with geometric '
                                    'quality; the effect depends on scene density and visibility.'},
 'wall_protrusion_distance': {'summary': 'Minimum separation distinguishing an object from wall '
                                         'thickness/noise.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'Avoids protecting wall noise but may miss objects close '
                                        'to walls.',
                              'lower': 'Protects close objects but may preserve thick-wall noise.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'wall_object_protection': {'summary': 'Strength of object evidence reducing the wall score.',
                            'detail': 'The effective value is recorded with the run. Compare '
                                      'identical inputs using a pinned baseline when changing this '
                                      'setting.',
                            'higher': 'Protects furniture near walls and preserves more wall '
                                      'junctions.',
                            'lower': 'More aggressive wall suppression with greater object loss '
                                     'risk.',
                            'cost': 'Inspect stage runtime and memory together with geometric '
                                    'quality; the effect depends on scene density and visibility.'},
 'wall_filter_enabled': {'summary': 'Apply scene-specific wall scores.',
                         'detail': 'The effective value is recorded with the run. Compare '
                                   'identical inputs using a pinned baseline when changing this '
                                   'setting.',
                         'higher': 'Enabled: apply budget penalties and configured hard rejection.',
                         'lower': 'Disabled: baseline without wall-score filtering.',
                         'cost': 'Inspect stage runtime and memory together with geometric '
                                 'quality; the effect depends on scene density and visibility.'},
 'wall_penalty_weight': {'summary': 'Soft penalty lowering wall keypoints in the final budget.',
                         'detail': 'The effective value is recorded with the run. Compare '
                                   'identical inputs using a pinned baseline when changing this '
                                   'setting.',
                         'higher': 'Walls consume fewer places; misclassified object keypoints can '
                                   'be displaced.',
                         'lower': 'Ranking stays closer to raw Harris response.',
                         'cost': 'Inspect stage runtime and memory together with geometric '
                                 'quality; the effect depends on scene density and visibility.'},
 'wall_hard_reject': {'summary': 'Allow removal of unambiguous wall keypoints.',
                      'detail': 'The effective value is recorded with the run. Compare identical '
                                'inputs using a pinned baseline when changing this setting.',
                      'higher': 'Enabled: actually reduces the feature count, even below the '
                                'budget cap.',
                      'lower': 'Disabled: wall scores affect ranking without this hard rejection.',
                      'cost': 'Inspect stage runtime and memory together with geometric quality; '
                              'the effect depends on scene density and visibility.'},
 'wall_reject_threshold': {'summary': 'Minimum wall score triggering hard rejection.',
                           'detail': 'The effective value is recorded with the run. Compare '
                                     'identical inputs using a pinned baseline when changing this '
                                     'setting.',
                           'higher': 'More conservative rejection and more residual wall points.',
                           'lower': 'Stronger wall removal with more risk to adjacent objects.',
                           'cost': 'Inspect stage runtime and memory together with geometric '
                                   'quality; the effect depends on scene density and visibility.'},
 'wall_reject_max_object_score': {'summary': 'Maximum object evidence allowed for wall rejection.',
                                  'detail': 'The effective value is recorded with the run. Compare '
                                            'identical inputs using a pinned baseline when '
                                            'changing this setting.',
                                  'higher': 'Allows rejection despite stronger object evidence.',
                                  'lower': 'Protects points even with weak protrusion evidence.',
                                  'cost': 'Inspect stage runtime and memory together with '
                                          'geometric quality; the effect depends on scene density '
                                          'and visibility.'},
 'wall_budget_enabled': {'summary': 'Limit the final budget share occupied by unprotected walls.',
                         'detail': 'The effective value is recorded with the run. Compare '
                                   'identical inputs using a pinned baseline when changing this '
                                   'setting.',
                         'higher': 'Enabled: prevent walls from dominating descriptors.',
                         'lower': 'Disabled: rely on response ranking and spatial balancing.',
                         'cost': 'Inspect stage runtime and memory together with geometric '
                                 'quality; the effect depends on scene density and visibility.'},
 'wall_budget_affinity_threshold': {'summary': 'Minimum affinity for counting a point in the wall '
                                               'quota.',
                                    'detail': 'The effective value is recorded with the run. '
                                              'Compare identical inputs using a pinned baseline '
                                              'when changing this setting.',
                                    'higher': 'Only clear wall points enter the quota; some noise '
                                              'escapes.',
                                    'lower': 'Includes noisy edges, with more risk to nearly '
                                             'coplanar objects.',
                                    'cost': 'Inspect stage runtime and memory together with '
                                            'geometric quality; the effect depends on scene '
                                            'density and visibility.'},
 'wall_budget_proximity_threshold': {'summary': 'Plane proximity criterion for quota assignment '
                                                'despite unstable normals.',
                                     'detail': 'The effective value is recorded with the run. '
                                               'Compare identical inputs using a pinned baseline '
                                               'when changing this setting.',
                                     'higher': 'Only points nearly on the plane enter the quota.',
                                     'lower': 'Includes broader wall boundaries and more '
                                              'adjacent-object risk.',
                                     'cost': 'Inspect stage runtime and memory together with '
                                             'geometric quality; the effect depends on scene '
                                             'density and visibility.'},
 'wall_budget_max_fraction': {'summary': 'Maximum budget fraction for unprotected wall points.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'Preserves architectural context, potentially dominating '
                                        'matching.',
                              'lower': 'Favors objects while reducing architectural anchors.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'wall_budget_cell_size': {'summary': 'Cell size distributing the retained wall quota.',
                           'detail': 'The effective value is recorded with the run. Compare '
                                     'identical inputs using a pinned baseline when changing this '
                                     'setting.',
                           'higher': 'Sparser wall coverage with less redundancy.',
                           'lower': 'More cells and possible wall candidates.',
                           'cost': 'Inspect stage runtime and memory together with geometric '
                                   'quality; the effect depends on scene density and visibility.'},
 'wall_budget_max_per_cell': {'summary': 'Maximum wall candidates per cell and plane.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'More nearby detail and redundancy.',
                              'lower': 'Fewer representatives per region.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'wall_budget_min_features': {'summary': 'Recall floor after the wall quota.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'Protects sparse scenes but allows more wall points.',
                              'lower': 'More aggressive cleanup; zero removes this floor.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'component_budget_enabled': {'summary': 'Adapt keypoint count to connected spatial regions.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'Enabled: reduce redundant regions while preserving '
                                        'separate objects.',
                              'lower': 'Disabled: use only the global cap.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'component_budget_radius': {'summary': 'Connection distance for regional keypoint budgeting.',
                             'detail': 'The effective value is recorded with the run. Compare '
                                       'identical inputs using a pinned baseline when changing '
                                       'this setting.',
                             'higher': 'Merges more structures; nearby objects can share a quota.',
                             'lower': 'Fragments objects and weakens regional capping.',
                             'cost': 'Inspect stage runtime and memory together with geometric '
                                     'quality; the effect depends on scene density and '
                                     'visibility.'},
 'component_budget_max_per_component': {'summary': 'Maximum retained keypoints in one connected '
                                                   'region.',
                                        'detail': 'The effective value is recorded with the run. '
                                                  'Compare identical inputs using a pinned '
                                                  'baseline when changing this setting.',
                                        'higher': 'More detail with slower, more redundant '
                                                  'matching.',
                                        'lower': 'Faster, with a risk of removing useful corners.',
                                        'cost': 'Inspect stage runtime and memory together with '
                                                'geometric quality; the effect depends on scene '
                                                'density and visibility.'},
 'component_budget_min_features': {'summary': 'Recall floor after regional budgeting.',
                                   'detail': 'The effective value is recorded with the run. '
                                             'Compare identical inputs using a pinned baseline '
                                             'when changing this setting.',
                                   'higher': 'Protects sparse scenes with more descriptors.',
                                   'lower': 'Stronger reduction on simple or fragmented scenes.',
                                   'cost': 'Inspect stage runtime and memory together with '
                                           'geometric quality; the effect depends on scene density '
                                           'and visibility.'},
 'floor_height': {'summary': 'Minimum keypoint height above estimated ground.',
                  'detail': 'The effective value is recorded with the run. Compare identical '
                            'inputs using a pinned baseline when changing this setting.',
                  'higher': 'Removes more ground but may remove furniture feet.',
                  'lower': 'Preserves low structures and more ground/wall junctions.',
                  'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                          'effect depends on scene density and visibility.'},
 'max_scan_keypoints': {'summary': 'Maximum scene keypoints passed to descriptors and matching.',
                        'detail': 'The effective value is recorded with the run. Compare identical '
                                  'inputs using a pinned baseline when changing this setting.',
                        'higher': 'Better secondary-object recall with greater descriptor and '
                                  'matching cost.',
                        'lower': 'Faster; a dominant object or wall may consume the budget.',
                        'cost': 'Inspect stage runtime and memory together with geometric quality; '
                                'the effect depends on scene density and visibility.'},
 'spatial_keypoint_balance': {'summary': 'Distribute the budget across 3D cells.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'Enabled: broader coverage, sometimes retaining weaker '
                                        'local points.',
                              'lower': 'Disabled: global response ranking can concentrate on one '
                                       'object.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'spatial_keypoint_cell': {'summary': 'Cell size used for spatial balancing.',
                           'detail': 'The effective value is recorded with the run. Compare '
                                     'identical inputs using a pinned baseline when changing this '
                                     'setting.',
                           'higher': 'Fewer, larger cells; less spatial diversity.',
                           'lower': 'More uniform coverage, but weak regions may receive places '
                                    'early.',
                           'cost': 'Inspect stage runtime and memory together with geometric '
                                   'quality; the effect depends on scene density and visibility.'},
 'min_2d_corner_fraction': {'summary': 'Minimum budget fraction reserved for 2D corners and thin '
                                       'structures.',
                            'detail': 'The effective value is recorded with the run. Compare '
                                      'identical inputs using a pinned baseline when changing this '
                                      'setting.',
                            'higher': 'Favors edges and tabletops at the expense of some 3D '
                                      'corners.',
                            'lower': 'Favors 3D Harris stability but may miss planar furniture '
                                     'details.',
                            'cost': 'Inspect stage runtime and memory together with geometric '
                                    'quality; the effect depends on scene density and visibility.'},
 'parallel_scenes': {'summary': 'Number of debug scenes computed concurrently.',
                     'detail': 'The effective value is recorded with the run. Compare identical '
                               'inputs using a pinned baseline when changing this setting.',
                     'higher': 'Less elapsed time when cores and RAM permit; more memory and '
                               'contention.',
                     'lower': 'Less contention and simpler profiling.',
                     'cost': 'Inspect stage runtime and memory together with geometric quality; '
                             'the effect depends on scene density and visibility.'},
 'voxel_size': {'summary': 'Spatial TSDF voxel size in meters.',
                'detail': 'The effective value is recorded with the run. Compare identical inputs '
                          'using a pinned baseline when changing this setting.',
                'higher': 'Lower memory and runtime, but thin structures can disappear.',
                'lower': 'More detail and much higher memory requirements.',
                'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                        'effect depends on scene density and visibility.'},
 'truncation': {'summary': 'TSDF update distance around measured surfaces.',
                'detail': 'The effective value is recorded with the run. Compare identical inputs '
                          'using a pinned baseline when changing this setting.',
                'higher': 'More noise tolerance but greater smoothing and nearby-surface '
                          'interaction.',
                'lower': 'Sharper localization with more holes under sparse/noisy observations.',
                'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                        'effect depends on scene density and visibility.'},
 'depth_trunc': {'summary': 'Maximum integrated depth in meters.',
                 'detail': 'The effective value is recorded with the run. Compare identical inputs '
                           'using a pinned baseline when changing this setting.',
                 'higher': 'More background and possibly larger bounds and noisier measurements.',
                 'lower': 'Focus on nearby geometry; distant objects can be cut off.',
                 'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                         'effect depends on scene density and visibility.'},
 'frame_stride': {'summary': 'Integrate at most one out of N frames, subject to min_frames.',
                  'detail': 'The effective value is recorded with the run. Compare identical '
                            'inputs using a pinned baseline when changing this setting.',
                  'higher': 'Faster fusion with fewer viewpoints and more holes.',
                  'lower': 'More viewpoints and potentially more complete geometry.',
                  'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                          'effect depends on scene density and visibility.'},
 'min_frames': {'summary': 'Minimum target viewpoint count; stride is reduced when possible.',
                'detail': 'The effective value is recorded with the run. Compare identical inputs '
                          'using a pinned baseline when changing this setting.',
                'higher': 'Protects thin or occluded surfaces in short sequences at extra cost.',
                'lower': 'Faster previews with less complete observation.',
                'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                        'effect depends on scene density and visibility.'},
 'max_frames': {'summary': 'Cap selected frames after stride selection. Zero means unlimited.',
                'detail': 'The effective value is recorded with the run. Compare identical inputs '
                          'using a pinned baseline when changing this setting.',
                'higher': 'Longer observed sequence and usually more visible object faces.',
                'lower': 'Faster but more partial geometry.',
                'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                        'effect depends on scene density and visibility.'},
 'iso_sampling': {'summary': 'Spatial downsampling size for extracted surface points.',
                  'detail': 'The effective value is recorded with the run. Compare identical '
                            'inputs using a pinned baseline when changing this setting.',
                  'higher': 'Smaller clouds with fewer thin-surface details.',
                  'lower': 'Denser surfaces and greater display/processing cost.',
                  'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                          'effect depends on scene density and visibility.'},
 'max_surface_points': {'summary': 'Point cap after spatial surface downsampling.',
                        'detail': 'The effective value is recorded with the run. Compare identical '
                                  'inputs using a pinned baseline when changing this setting.',
                        'higher': 'Preserves more rare details when the cap is reached.',
                        'lower': 'Faster downstream processing but possible loss of small '
                                 'structures.',
                        'cost': 'Inspect stage runtime and memory together with geometric quality; '
                                'the effect depends on scene density and visibility.'},
 'surface_field': {'summary': 'Field used to extract geometry: raw or smoothed.',
                   'detail': 'The effective value is recorded with the run. Compare identical '
                             'inputs using a pinned baseline when changing this setting.',
                   'higher': 'Smoothed suppresses noise but may erase one- or two-voxel '
                             'structures.',
                   'lower': 'Raw preserves detail; normals can still use the smoothed field.',
                   'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                           'effect depends on scene density and visibility.'},
 'surface_min_support': {'summary': 'Minimum observed support around zero crossings.',
                         'detail': 'The effective value is recorded with the run. Compare '
                                   'identical inputs using a pinned baseline when changing this '
                                   'setting.',
                         'higher': 'Rejects weakly observed surfaces and may lose occluded legs.',
                         'lower': 'Recovers partial detail with more fragments near unknown space.',
                         'cost': 'Inspect stage runtime and memory together with geometric '
                                 'quality; the effect depends on scene density and visibility.'},
 'surface_extraction': {'summary': 'Choose strict zero crossings or controlled near-zero recovery.',
                        'detail': 'The effective value is recorded with the run. Compare identical '
                                  'inputs using a pinned baseline when changing this setting.',
                        'higher': 'Hybrid restores thin details alongside the strict surface.',
                        'lower': 'Zero-crossing is conservative and can miss surfaces without a '
                                 'sign change.',
                        'cost': 'Inspect stage runtime and memory together with geometric quality; '
                                'the effect depends on scene density and visibility.'},
 'surface_band_factor': {'summary': 'TSDF band width considered for recovering thin detail.',
                         'detail': 'The effective value is recorded with the run. Compare '
                                   'identical inputs using a pinned baseline when changing this '
                                   'setting.',
                         'higher': 'More candidates, noise and processing.',
                         'lower': 'Closer to the surface, potentially missing thin structures.',
                         'cost': 'Inspect stage runtime and memory together with geometric '
                                 'quality; the effect depends on scene density and visibility.'},
 'surface_rescue_distance_factor': {'summary': 'Minimum separation from strict surface for added '
                                               'recovery points.',
                                    'detail': 'The effective value is recorded with the run. '
                                              'Compare identical inputs using a pinned baseline '
                                              'when changing this setting.',
                                    'higher': 'Avoids duplicates and thick walls but recovers '
                                              'fewer nearby details.',
                                    'lower': 'Adds more points near existing surfaces.',
                                    'cost': 'Inspect stage runtime and memory together with '
                                            'geometric quality; the effect depends on scene '
                                            'density and visibility.'},
 'surface_rescue_min_weight': {'summary': 'Minimum observation weight for near-zero recovery.',
                               'detail': 'The effective value is recorded with the run. Compare '
                                         'identical inputs using a pinned baseline when changing '
                                         'this setting.',
                               'higher': 'Less noise but fewer rarely observed faces.',
                               'lower': 'More weak observations and false fragments.',
                               'cost': 'Inspect stage runtime and memory together with geometric '
                                       'quality; the effect depends on scene density and '
                                       'visibility.'},
 'gradient_smoothing_sigma': {'summary': 'Smoothing of the field used for normals.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'More stable normals, but nearby structures can mix.',
                              'lower': 'More local normals with greater sensor-noise sensitivity.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'depth_edge_threshold': {'summary': 'Depth jump identifying foreground/background boundaries.',
                          'detail': 'The effective value is recorded with the run. Compare '
                                    'identical inputs using a pinned baseline when changing this '
                                    'setting.',
                          'higher': 'Protects only strong edges and permits more leg/ground '
                                    'mixing.',
                          'lower': 'Protects more details but may flag slanted surfaces.',
                          'cost': 'Inspect stage runtime and memory together with geometric '
                                  'quality; the effect depends on scene density and visibility.'},
 'depth_edge_background_weight': {'summary': 'Background contribution immediately behind a '
                                             'foreground silhouette.',
                                  'detail': 'The effective value is recorded with the run. Compare '
                                            'identical inputs using a pinned baseline when '
                                            'changing this setting.',
                                  'higher': 'Closer to ordinary TSDF fusion, risking '
                                            'thin-structure loss.',
                                  'lower': 'Protects foreground, but an erroneous near measurement '
                                           'may persist.',
                                  'cost': 'Inspect stage runtime and memory together with '
                                          'geometric quality; the effect depends on scene density '
                                          'and visibility.'},
 'depth_edge_radius': {'summary': 'Pixel radius of foreground/background protection.',
                       'detail': 'The effective value is recorded with the run. Compare identical '
                                 'inputs using a pinned baseline when changing this setting.',
                       'higher': 'Wider protection and broader influence.',
                       'lower': 'More local intervention, possibly insufficient for distant legs.',
                       'cost': 'Inspect stage runtime and memory together with geometric quality; '
                               'the effect depends on scene density and visibility.'},
 'volume_layout': {'summary': 'Dense grid or voxel-hashed sparse blocks allocated near '
                              'observations.',
                   'detail': 'The effective value is recorded with the run. Compare identical '
                             'inputs using a pinned baseline when changing this setting.',
                   'higher': 'Sparse supports larger, finer scenes; dense stores the entire '
                             'bounding box.',
                   'lower': 'Auto chooses from hardware and estimated volume size.',
                   'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                           'effect depends on scene density and visibility.'},
 'sparse_block_resolution': {'summary': 'Voxels per side of a sparse block.',
                             'detail': 'The effective value is recorded with the run. Compare '
                                       'identical inputs using a pinned baseline when changing '
                                       'this setting.',
                             'higher': 'Fewer hash keys but more extra voxels per touched region.',
                             'lower': 'Tighter allocation with more blocks and hash overhead.',
                             'cost': 'Inspect stage runtime and memory together with geometric '
                                     'quality; the effect depends on scene density and '
                                     'visibility.'},
 'sparse_block_count': {'summary': 'Reserved active sparse-block capacity.',
                        'detail': 'The effective value is recorded with the run. Compare identical '
                                  'inputs using a pinned baseline when changing this setting.',
                        'higher': 'Supports larger/finer scenes with greater memory reservation.',
                        'lower': 'Lower reservation with a risk of insufficient capacity.',
                        'cost': 'Inspect stage runtime and memory together with geometric quality; '
                                'the effect depends on scene density and visibility.'},
 'visibility_depth_stride': {'summary': 'Downsampling stride for stored visibility depth images.',
                             'detail': 'The effective value is recorded with the run. Compare '
                                       'identical inputs using a pinned baseline when changing '
                                       'this setting.',
                             'higher': 'Lower memory and faster queries with less precise '
                                       'silhouettes.',
                             'lower': 'More precise visibility near thin parts with greater memory '
                                      'use.',
                             'cost': 'Inspect stage runtime and memory together with geometric '
                                     'quality; the effect depends on scene density and '
                                     'visibility.'},
 'visibility_surface_band_factor': {'summary': 'Thickness of the occupied band around measured '
                                               'depth.',
                                    'detail': 'The effective value is recorded with the run. '
                                              'Compare identical inputs using a pinned baseline '
                                              'when changing this setting.',
                                    'higher': 'More noise tolerance but thicker, less distinctive '
                                              'occupancy.',
                                    'lower': 'Sharper occupancy, possibly missing poorly aligned '
                                             'surfaces.',
                                    'cost': 'Inspect stage runtime and memory together with '
                                            'geometric quality; the effect depends on scene '
                                            'density and visibility.'},
 'backend': {'summary': 'Compute engine for the selected volume layout.',
             'detail': 'The effective value is recorded with the run. Compare identical inputs '
                       'using a pinned baseline when changing this setting.',
             'higher': 'CUDA can accelerate compatible operations when hardware is available.',
             'lower': 'Auto falls back to CPU when CUDA is unavailable.',
             'cost': 'Inspect stage runtime and memory together with geometric quality; the effect '
                     'depends on scene density and visibility.'},
 'occ_grid_res': {'summary': 'Cells per axis in the local scan occupancy grid.',
                  'detail': 'The effective value is recorded with the run. Compare identical '
                            'inputs using a pinned baseline when changing this setting.',
                  'higher': 'Finer local geometry; grid size grows cubically.',
                  'lower': 'Faster but merges local details.',
                  'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                          'effect depends on scene density and visibility.'},
 'occ_extent': {'summary': 'Spatial half-extent covered by the local descriptor.',
                'detail': 'The effective value is recorded with the run. Compare identical inputs '
                          'using a pinned baseline when changing this setting.',
                'higher': 'More context, possibly mixing object, wall and neighboring furniture.',
                'lower': 'Isolates fine detail but loses wider distinctive structure.',
                'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                        'effect depends on scene density and visibility.'},
 'distance_unit': {'summary': 'Metric normalization applied before the UDF exponent. Zero uses '
                              'automatic cell size.',
                   'detail': 'The effective value is recorded with the run. Compare identical '
                             'inputs using a pinned baseline when changing this setting.',
                   'higher': 'Smaller computed distances and more accepted correspondences.',
                   'lower': 'Larger distances and stricter matching.',
                   'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                           'effect depends on scene density and visibility.'},
 'distance_exponent': {'summary': 'Exponent emphasizing distances from occupied scan cells to '
                                  'model surface.',
                       'detail': 'The effective value is recorded with the run. Compare identical '
                                 'inputs using a pinned baseline when changing this setting.',
                       'higher': 'Stronger penalties for large mismatches, with greater noise '
                                 'sensitivity.',
                       'lower': 'More tolerance but less distinction between similar forms.',
                       'cost': 'Inspect stage runtime and memory together with geometric quality; '
                               'the effect depends on scene density and visibility.'},
 'max_occupied': {'summary': 'Maximum occupied cells sampled for descriptor comparison.',
                  'detail': 'The effective value is recorded with the run. Compare identical '
                            'inputs using a pinned baseline when changing this setting.',
                  'higher': 'Less sampling error and greater comparison cost.',
                  'lower': 'Faster with noisier distance estimates.',
                  'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                          'effect depends on scene density and visibility.'},
 'utility_candidate_models': {'summary': 'Model-pool size used only for descriptor diagnostics.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'More realistic ambiguity estimates with greater '
                                        'diagnostic cost.',
                              'lower': 'Fast preview that may overestimate distinctiveness.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'utility_model_features': {'summary': 'Maximum model keypoints used in diagnostics.',
                            'detail': 'The effective value is recorded with the run. Compare '
                                      'identical inputs using a pinned baseline when changing this '
                                      'setting.',
                            'higher': 'More local correspondence recall and greater cost.',
                            'lower': 'Faster but may miss distinctive model parts.',
                            'cost': 'Inspect stage runtime and memory together with geometric '
                                    'quality; the effect depends on scene density and visibility.'},
 'utility_rotations': {'summary': 'Orientations tested per diagnostic local pair.',
                       'detail': 'The effective value is recorded with the run. Compare identical '
                                 'inputs using a pinned baseline when changing this setting.',
                       'higher': 'Fewer missed alignments and greater cost.',
                       'lower': 'Faster but may miss the correct orientation.',
                       'cost': 'Inspect stage runtime and memory together with geometric quality; '
                               'the effect depends on scene density and visibility.'},
 'utility_distance_threshold': {'summary': 'Maximum distance counting a model as compatible.',
                                'detail': 'The effective value is recorded with the run. Compare '
                                          'identical inputs using a pinned baseline when changing '
                                          'this setting.',
                                'higher': 'Greater recall and more ambiguous generic surfaces.',
                                'lower': 'Closer local shapes required; more unmatched '
                                         'descriptors.',
                                'cost': 'Inspect stage runtime and memory together with geometric '
                                        'quality; the effect depends on scene density and '
                                        'visibility.'},
 'descriptor_ratio_threshold': {'summary': 'Best/second descriptor ratio threshold. Zero disables '
                                           'the ratio test.',
                                'detail': 'The effective value is recorded with the run. Compare '
                                          'identical inputs using a pinned baseline when changing '
                                          'this setting.',
                                'higher': 'More correspondences, including ambiguous ones.',
                                'lower': 'More distinctive matches but reduced recall on repeated '
                                         'forms.',
                                'cost': 'Inspect stage runtime and memory together with geometric '
                                        'quality; the effect depends on scene density and '
                                        'visibility.'},
 'utility_margin_threshold': {'summary': 'Minimum relative gap between the first and second model.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'More descriptors classified as ambiguous.',
                              'lower': 'Less stable rankings accepted as informative.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'utility_max_valid_models': {'summary': 'Maximum compatible models still considered specific.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'Allows more category-common descriptors.',
                              'lower': 'Favors rare details but may penalize useful repeated '
                                       'structures.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'utility_entropy_threshold': {'summary': 'Maximum compatible-score dispersion for an informative '
                                          'descriptor.',
                               'detail': 'The effective value is recorded with the run. Compare '
                                         'identical inputs using a pinned baseline when changing '
                                         'this setting.',
                               'higher': 'Tolerates more ambiguity.',
                               'lower': 'Requires a smaller set of models to dominate.',
                               'cost': 'Inspect stage runtime and memory together with geometric '
                                       'quality; the effect depends on scene density and '
                                       'visibility.'},
 'candidate_top_k': {'summary': 'Final shortlist size after reranking. Per-group budgets can '
                                'supersede the global cap.',
                     'detail': 'The effective value is recorded with the run. Compare identical '
                               'inputs using a pinned baseline when changing this setting.',
                     'higher': 'Better recall at substantially higher matching cost.',
                     'lower': 'Faster matching, with possible loss of the correct model.',
                     'cost': 'Inspect stage runtime and memory together with geometric quality; '
                             'the effect depends on scene density and visibility.'},
 'candidate_pool_k': {'summary': 'Broad global pool size before local reranking.',
                      'detail': 'The effective value is recorded with the run. Compare identical '
                                'inputs using a pinned baseline when changing this setting.',
                      'higher': 'More chances for a correct model to reach local scoring.',
                      'lower': 'Faster querying with stronger global-statistics bias.',
                      'cost': 'Inspect stage runtime and memory together with geometric quality; '
                              'the effect depends on scene density and visibility.'},
 'candidate_diverse': {'summary': 'Reserve part of preselection for geometric diversity.',
                       'detail': 'The effective value is recorded with the run. Compare identical '
                                 'inputs using a pinned baseline when changing this setting.',
                       'higher': 'Enabled: more unusual shape families can survive.',
                       'lower': 'Disabled: strict ranking may be dominated by similar variants.',
                       'cost': 'Inspect stage runtime and memory together with geometric quality; '
                               'the effect depends on scene density and visibility.'},
 'candidate_diversity': {'summary': 'Fraction reserved for diversity within the pool.',
                         'detail': 'The effective value is recorded with the run. Compare '
                                   'identical inputs using a pinned baseline when changing this '
                                   'setting.',
                         'higher': 'Broader shape exploration.',
                         'lower': 'More weight on the best global scores.',
                         'cost': 'Inspect stage runtime and memory together with geometric '
                                 'quality; the effect depends on scene density and visibility.'},
 'local_rerank': {'summary': 'Rerank the broad pool with local scene descriptors.',
                  'detail': 'The effective value is recorded with the run. Compare identical '
                            'inputs using a pinned baseline when changing this setting.',
                  'higher': 'Enabled: local geometry influences ranking at extra cost.',
                  'lower': 'Disabled: faster but more dependent on global shape statistics.',
                  'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                          'effect depends on scene density and visibility.'},
 'grouped_query': {'summary': 'Query independently for each object group and merge rankings.',
                   'detail': 'The effective value is recorded with the run. Compare identical '
                             'inputs using a pinned baseline when changing this setting.',
                   'higher': 'Enabled: more object-specific proposals and group provenance.',
                   'lower': 'Disabled: use one scene-wide query.',
                   'cost': 'Inspect stage runtime and memory together with geometric quality; the '
                           'effect depends on scene density and visibility.'},
 'candidate_local_extra_k': {'summary': 'Additional candidates from the compact local UDF index. '
                                        'Preserves the original pool and requires '
                                        'local_candidate_index.',
                             'detail': 'The effective value is recorded with the run. Compare '
                                       'identical inputs using a pinned baseline when changing '
                                       'this setting.',
                             'higher': 'Explore more models and increase reranking work.',
                             'lower': 'Smaller supplement; zero disables this route.',
                             'cost': 'Inspect stage runtime and memory together with geometric '
                                     'quality; the effect depends on scene density and '
                                     'visibility.'},
 'candidate_max_groups': {'summary': 'Maximum originating object groups scored per model. Zero '
                                     'tests all originating groups.',
                          'detail': 'The effective value is recorded with the run. Compare '
                                    'identical inputs using a pinned baseline when changing this '
                                    'setting.',
                          'higher': 'A larger positive limit preserves ambiguous associations at '
                                    'extra cost.',
                          'lower': 'A small positive limit is faster; zero is unlimited.',
                          'cost': 'Inspect stage runtime and memory together with geometric '
                                  'quality; the effect depends on scene density and visibility.'},
 'candidate_min_group_descriptors': {'summary': 'Minimum descriptors required to issue an object '
                                                'query.',
                                     'detail': 'The effective value is recorded with the run. '
                                               'Compare identical inputs using a pinned baseline '
                                               'when changing this setting.',
                                     'higher': 'Rejects small accidental fragments but may miss '
                                               'poorly observed objects.',
                                     'lower': 'Preserves partial objects with less stable ranking '
                                              'and more work.',
                                     'cost': 'Inspect stage runtime and memory together with '
                                             'geometric quality; the effect depends on scene '
                                             'density and visibility.'},
 'candidate_group_weighting': {'summary': 'Weight group proposals by descriptor count and '
                                          'geometric support.',
                               'detail': 'The effective value is recorded with the run. Compare '
                                         'identical inputs using a pinned baseline when changing '
                                         'this setting.',
                               'higher': 'Enabled: reliable groups have more influence.',
                               'lower': 'Disabled: all groups vote equally.',
                               'cost': 'Inspect stage runtime and memory together with geometric '
                                       'quality; the effect depends on scene density and '
                                       'visibility.'},
 'candidate_pool_min_per_group': {'summary': 'Protected broad-pool places for each object group.',
                                  'detail': 'The effective value is recorded with the run. Compare '
                                            'identical inputs using a pinned baseline when '
                                            'changing this setting.',
                                  'higher': 'Better object coverage, including potentially weak '
                                            'groups.',
                                  'lower': 'Merged scores dominate; an object can disappear.',
                                  'cost': 'Inspect stage runtime and memory together with '
                                          'geometric quality; the effect depends on scene density '
                                          'and visibility.'},
 'candidate_global_pool_fraction': {'summary': 'Reserve pool places for the whole-scene ranking.',
                                    'detail': 'The effective value is recorded with the run. '
                                              'Compare identical inputs using a pinned baseline '
                                              'when changing this setting.',
                                    'higher': 'Protects recall when grouping loses useful object '
                                              'parts.',
                                    'lower': 'Favors specialized group proposals.',
                                    'cost': 'Inspect stage runtime and memory together with '
                                            'geometric quality; the effect depends on scene '
                                            'density and visibility.'},
 'candidate_top_min_per_group': {'summary': 'Protected final Top-k places per object group.',
                                 'detail': 'The effective value is recorded with the run. Compare '
                                           'identical inputs using a pinned baseline when changing '
                                           'this setting.',
                                 'higher': 'Protects object recall but retains more weak '
                                           'hypotheses.',
                                 'lower': 'Higher overall selectivity with risk of losing an '
                                          'object.',
                                 'cost': 'Inspect stage runtime and memory together with geometric '
                                         'quality; the effect depends on scene density and '
                                         'visibility.'},
 'candidate_descriptor_ratio_threshold': {'summary': 'Best/second local descriptor ratio in '
                                                     'reranking. Zero disables.',
                                          'detail': 'The effective value is recorded with the run. '
                                                    'Compare identical inputs using a pinned '
                                                    'baseline when changing this setting.',
                                          'higher': 'Retains more potentially ambiguous '
                                                    'correspondences.',
                                          'lower': 'Requires more distinct descriptors and may '
                                                   'reduce recall.',
                                          'cost': 'Inspect stage runtime and memory together with '
                                                  'geometric quality; the effect depends on scene '
                                                  'density and visibility.'},
 'candidate_local_weight': {'summary': 'Balance local correspondence evidence against initial '
                                       'group rank.',
                            'detail': 'The effective value is recorded with the run. Compare '
                                      'identical inputs using a pinned baseline when changing this '
                                      'setting.',
                            'higher': 'Favors local correspondence quality and spread.',
                            'lower': 'Preserves more of the broad-pool order.',
                            'cost': 'Inspect stage runtime and memory together with geometric '
                                    'quality; the effect depends on scene density and visibility.'},
 'candidate_extent_weight': {'summary': 'Weight of observed-group versus model extent '
                                        'compatibility.',
                             'detail': 'The effective value is recorded with the run. Compare '
                                       'identical inputs using a pinned baseline when changing '
                                       'this setting.',
                             'higher': 'Rejects strongly mismatched object proportions.',
                             'lower': 'Relies more on local details under heavy partial '
                                      'observation.',
                             'cost': 'Inspect stage runtime and memory together with geometric '
                                     'quality; the effect depends on scene density and '
                                     'visibility.'},
 'candidate_extent_quota_fraction': {'summary': 'Top-k fraction protected by observed object '
                                                'proportions.',
                                     'detail': 'The effective value is recorded with the run. '
                                               'Compare identical inputs using a pinned baseline '
                                               'when changing this setting.',
                                     'higher': 'More dimension-compatible candidates, including '
                                               'simple shapes.',
                                     'lower': 'Local score and preselection prior dominate.',
                                     'cost': 'Inspect stage runtime and memory together with '
                                             'geometric quality; the effect depends on scene '
                                             'density and visibility.'},
 'candidate_target_matches': {'summary': 'Distinct local matches required for full support weight.',
                              'detail': 'The effective value is recorded with the run. Compare '
                                        'identical inputs using a pinned baseline when changing '
                                        'this setting.',
                              'higher': 'Penalizes models supported by few points.',
                              'lower': 'Supports partial objects but permits more coincidences.',
                              'cost': 'Inspect stage runtime and memory together with geometric '
                                      'quality; the effect depends on scene density and '
                                      'visibility.'},
 'candidate_target_coverage': {'summary': 'Local descriptor coverage needed for full support '
                                          'weight.',
                               'detail': 'The effective value is recorded with the run. Compare '
                                         'identical inputs using a pinned baseline when changing '
                                         'this setting.',
                               'higher': 'Requires a fuller explanation of the observation.',
                               'lower': 'Tolerates partial and occluded scans.',
                               'cost': 'Inspect stage runtime and memory together with geometric '
                                       'quality; the effect depends on scene density and '
                                       'visibility.'},
 'surface_preview_points': {'summary': 'Maximum scan points sent to the 3D preview; does not '
                                       'affect retrieval.',
                            'detail': 'The effective value is recorded with the run. Compare '
                                      'identical inputs using a pinned baseline when changing this '
                                      'setting.',
                            'higher': 'Denser, more informative visualization and larger payloads.',
                            'lower': 'Faster display and JSON transfer.',
                            'cost': 'Inspect stage runtime and memory together with geometric '
                                    'quality; the effect depends on scene density and visibility.'}}

def english_parameter_help(stage):
    """Return independent help entries; the caller selects its stage schema."""
    return {key: dict(value) for key, value in PARAMETER_HELP.items()}
