'\nCentral configuration for the ObjectSensing reproduction and subsequent experiments. Values include paper-inspired parameters and implementation-specific settings; they are not all original-paper defaults. See docs/PAPER_MAPPING.md and docs/PARAMETERS.md.\n'

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SDFConfig:
    'TSDF volumetric fusion (paper section 4 / VoxelHashing).'
    voxel_size: float = 0.015          # 1,5 cm avec voxel hashing sparse sur CPU
    truncation: float = 0.06           # bande de troncature du TSDF (4 voxels)
    depth_trunc: float = 4.5           # profondeur max prise en compte (m)
    depth_scale: float = 1000.0        # depth.png en mm -> m
    max_frames: int = 0                # 0 = toutes the frames
    frame_stride: int = 6              # profil commun Office / IKEA / Single Chair
    min_frames: int = 50               # plancher adaptatif sur the scènes courtes
    iso_sampling: float = 0.01         # pas d'échantillonnage de l'iso-surface (m)
    max_surface_points: int = 175000   # compromis densité/détails validated sur trois scènes
    surface_field: str = "raw"          # raw préserve the détails ; smoothed = ancien comportement
    surface_min_support: float = 0.20   # support observé minimal aux passages par zéro
    surface_extraction: str = "hybrid"  # zero_crossing | hybrid | near_zero
    surface_band_factor: float = 1.5     # bande de secours autour de zéro, en voxels
    surface_rescue_distance_factor: float = 1.0  # difference minimal à l'iso-surface, en voxels
    surface_rescue_min_weight: float = 5.0  # observations minimales des détails récupérés
    gradient_smoothing_sigma: float = 0.8  # lissage du SDF réservé aux normales
    depth_edge_threshold: float = 0.05  # saut de profondeur traité comme silhouette (m)
    depth_edge_background_weight: float = 0.10  # poids du fond près d'un avant-plan
    depth_edge_radius: int = 1          # voisinage de protection des silhouettes (pixels)
    backend: str = "auto"              # auto | cpu | cuda (CuPy si disponible)
    volume_layout: str = "auto"         # auto | dense | sparse (voxel hashing Open3D)
    sparse_block_resolution: int = 16   # voxels par côté d'un bloc actif
    sparse_block_count: int = 50000     # capacité maximale du hash de blocs
    visibility_depth_stride: int = 2    # résolution des observations de visibilité
    visibility_surface_band_factor: float = 0.75  # demi-épaisseur occupée en voxels


@dataclass
class KeypointConfig:
    'Keypoint detection (paper section 4).'
    neighbor_radius: float = 0.05      # rayon de voisinage r pour PCA/Harris/2D
    curvature_threshold: float = 0.05  # t_ci : on échantillonne the points non-plans (c_i > t_ci)
    harris_k: float = 0.04             # R = det(C) - k*trace(C)^2
    harris_threshold: float = 0.008    # t_Ri
    harris_reference_neighbors: float = 6.0  # densité canonique du tenseur de normales
    convex_hull_ratio: float = math.pi / 3.0  # test coin 2D : aire < pi*r^2/3 (papier, sec.4)
    corner_plane_radius_factor: float = 3.0  # support étendu pour vérifier le « large plane »
    corner_plane_min_area: float = 0.02  # aire minimale du support planaire étendu (m²)
    corner_plane_normal_cos: float = 0.9  # cohérence minimale des normales du support
    nms_radius: float = 0.05           # rayon de non-maximum suppression
    adjust_iterations: int = 5         # ajustement itératif de position
    jitter_reject: float = None        # |p_final - p_orig| > r  -> rejet (None => r)
    dedup_radius: float = 0.01         # fusion des doublons après ajustement
    # Rejet scan-only très conservateur avant Harris. Il économise le calcul
    # des candidats placés sur une definite free/unknown boundary, mais reste
    # inactif pour the models ShapeNet qui ne possèdent pas de visibilité.
    pre_harris_visibility_filter_enabled: bool = True
    pre_harris_visibility_probe_radius: float = 0.03
    pre_harris_visibility_max_fraction: float = 0.58
    # Analyse scan-only des grands plans verticaux. Ces paramètres n'affectent
    # jamais the keypoints prétraités des models ShapeNet.
    wall_vertical_normal_cos: float = 0.30  # |n.up| maximal pour une normale murale
    wall_plane_normal_cos: float = 0.94     # cohérence des normales d'un même mur
    wall_plane_angle_degrees: float = 10.0  # largeur des groupes d'orientation
    wall_plane_distance: float = 0.06       # tolérance point-plan (m)
    wall_plane_min_points: int = 120        # support minimal après raffinement
    wall_plane_min_width: float = 0.80      # étendue horizontale minimale (m)
    wall_plane_min_height: float = 0.80     # étendue verticale minimale (m)
    wall_plane_min_area: float = 0.70       # aire rectangulaire minimale (m²)
    wall_plane_max_count: int = 6           # nombre maximal de murs dominants
    wall_plane_sample_points: int = 30000   # plafond du vote d'hypothèses
    wall_small_radius: float = 0.08          # support planaire local (m)
    wall_large_radius: float = 0.28          # support planaire contextuel (m)
    wall_protrusion_radius: float = 0.25     # recherche d'objet devant le mur (m)
    wall_protrusion_distance: float = 0.08   # séparation mur/objet significative (m)
    wall_object_protection: float = 0.70     # réduction du score mural si objet présent
    wall_filter_enabled: bool = True         # analyse murale active sur the scans
    wall_penalty_weight: float = 0.12        # pénalité du score de budget final
    wall_hard_reject: bool = True            # rejet des cas muraux non ambigus
    wall_reject_threshold: float = 0.20      # score mural minimal pour rejet
    wall_reject_max_object_score: float = 0.35  # preuve d'objet maximale au rejet
    wall_budget_enabled: bool = True          # limit the murs dans le budget final
    wall_budget_affinity_threshold: float = 0.05  # affinité minimale au plan
    wall_budget_proximity_threshold: float = 0.05  # proximité sans contrainte normale
    wall_budget_max_fraction: float = 0.05    # part maximale du budget pour the murs
    wall_budget_cell_size: float = 0.40       # cellule de répartition sur un mur (m)
    wall_budget_max_per_cell: int = 1         # points muraux admis par cellule/plan
    wall_budget_min_features: int = 40        # plancher de rappel après quota mural
    component_budget_enabled: bool = True     # plafond adaptatif par région spatiale
    component_budget_radius: float = 0.20     # rayon de connexité des régions (m)
    component_budget_max_per_component: int = 64  # points max par région
    component_budget_min_features: int = 40   # plancher global de rappel
    # Filtres scan-only appliqués après Harris et le filtre mural. Ils retirent
    # the maxima produits par une surface plane bruitée ou un bord RGB-D non
    # confirmé, puis exigent une response Harris stable à plus grande scale.
    planar_filter_enabled: bool = True
    planar_filter_radius: float = 0.12
    planar_filter_max_residual: float = 0.012
    planar_filter_normal_cos: float = 0.94
    planar_filter_angular_coverage: float = 0.70
    planar_filter_min_neighbors: int = 24
    hole_boundary_filter_enabled: bool = True
    hole_boundary_probe_radius: float = 0.03
    hole_boundary_max_fraction: float = 0.20
    repeatability_filter_enabled: bool = True
    repeatability_radius_factor: float = 1.6
    repeatability_response_ratio: float = 0.50
    quality_response_ratio: float = 0.35
    quality_score_radius: float = 0.45
    geometric_nms_radius: float = 0.10
    # Le plancher absolu historique reste un plafond de compatibilité. Le
    # plancher effectif est adaptatif et ne peut réintroduire qu'une fraction
    # limitée des candidats admissibles rejetés par un filtre doux.
    quality_filter_min_features: int = 80
    quality_filter_min_absolute: int = 20
    quality_filter_min_ratio: float = 0.18
    quality_filter_max_reintroduced_fraction: float = 0.20


@dataclass
class PrimitiveConfig:
    """Global support primitives: planes and lines (Section 5.1)."""
    plane_min_area: float = 0.04       # A* : aire min des plans (m^2)
    line_min_length: float = 0.2       # L* : longueur min des lignes (m)
    plane_grow_dist: float = 0.02      # distance d'inlier pour la croissance de plan
    plane_grow_normal: float = 0.9     # cos d'angle min entre normales (croissance)
    horizontal_cos: float = 0.92       # |n . up| > seuil  -> plan horizontal
    vertical_cos: float = 0.30         # |n . up| < seuil  -> plan/line vertical


@dataclass
class DescriptorConfig:
    """Local descriptors (sec. 5.1)."""
    udf_grid_res: int = 16             # résolution de la grille de la fonction de distance locale
    udf_extent: float = 0.12           # demi-étendue de la grille locale (m)
    occ_grid_res: int = 16             # résolution de la grille d'occupation (scan)
    occ_extent: float = 0.12
    # <= 0 choisit automatiquement la taille d'une cellule UDF. Les seuils
    # t_desc du papier s'appliquent ainsi à des distances sans dimension, et
    # non à des mètres élevés à la puissance alpha.
    distance_unit: float = 0.0         # unité de D_L (m), auto si <= 0
    distance_exponent: float = 4.0     # alpha (eq. 3)
    max_occupied: int = 150            # plafond de cellules occupées utilisées (perf)
    filter_hole_boundaries: bool = True  # ignore the surfaces vues seulement au bord d'un trou RGB-D


@dataclass
class MatchingConfig:
    """Matching de key points (sec. 5.2) + filtres."""
    n_uniform_rotations: int = 36      # rotations horizontales testées si pas de direction
    primitive_rotation_fallback: bool = False  # option de diagnostic; peut rendre une pose symétrique ambiguë
    confidence_threshold: float = 0.8  # t_c (eq. 5)
    size_ratio_reject: float = 3.0     # rejet si un ratio du vecteur 6D > 3
    scale_min: float = 1.0 / 1.5       # plage d'scale admissible (hauteur)
    scale_max: float = 1.5
    max_correspondences_per_kp: int = 8  # shortlist par key point de scan avant RANSAC
    # Le ratio reste optionnel : the probes Office montrent qu'imposer un seul
    # voisin local détruit the alternatives nécessaires au RANSAC géométrique.
    descriptor_ratio_threshold: float = 0.0  # meilleur/second ; 0 désactive
    # Le reranking doit rester comparable entre the expériences de RANSAC.
    # Sinon relever t_desc change simultanément le top-k et le matching, ce qui
    # empêche d'attribuer une régression à l'une des deux stages.
    # <= 0 réutilise le seuil du matching complet. La valeur explicite garde le
    # reranking comparable entre expériences de RANSAC.
    candidate_descriptor_distance_threshold: float = 128.0
    grouped_candidate_query: bool = True
    candidate_min_group_descriptors: int = 6
    candidate_pool_min_per_group: int = 20
    candidate_top_min_per_group: int = 6
    # Budget final indépendant par groupe d'objet. Une valeur positive remplace
    # le quota minimal suivi d'un remplissage global par l'union des top-k de
    # chaque groupe, afin qu'un groupe dominant n'évince pas the autres objets.
    candidate_top_per_group: int = 50
    candidate_global_pool_fraction: float = 0.70
    candidate_group_local_weight: float = 0.65
    candidate_local_extra_k: int = 64  # pool supplémentaire UDF, si l'index local existe
    group_pose_seed_enabled: bool = True
    group_pose_seed_count: int = 6
    require_constellation_support: bool = True  # final acceptance excludes unconfirmed shape-only seeds
    # Plafonds de performance (on garde the key points de plus forte response) :
    max_scan_keypoints: int = 400      # compromis multi-scènes mesuré
    max_model_keypoints: int = 120     # key points de modèle retenus
    spatial_keypoint_balance: bool = False  # répartit le budget entre cellules 3D
    spatial_keypoint_cell: float = 0.50     # taille des cellules d'équilibrage (m)
    min_2d_corner_fraction: float = 0.25  # réserve du plafond perf aux silhouettes/structures fines
    # Filtre intrinsèque peu coûteux une fois the descripteurs construits.
    descriptor_keypoint_filter_enabled: bool = True
    descriptor_min_occupied: int = 8
    descriptor_max_hole_fraction: float = 0.35
    descriptor_keypoint_min_features: int = 80
    descriptor_keypoint_min_absolute: int = 20
    descriptor_keypoint_min_ratio: float = 0.18
    descriptor_keypoint_max_reintroduced_fraction: float = 0.20
    quality_nms_radius: float = 0.08
    quality_min_score_ratio: float = 0.25
    floor_height: float = 0.08         # on ignore the key points de scan sous cette
                                       # hauteur au-dessus du sol (key points de sol)
    # Verification (sec.6) : score = couverture AVANT (modèle expliqué par le scan,
    # métrique du papier) ; la couverture ARRIÈRE sert seulement de garde-fou pour
    # rejeter the models "plaqués" sur un mur/sol (sinon min(av,ar) pénalise the
    # objets ajourés comme the chaises au profit des objets pleins type table).
    reverse_gate: float = 0.42         # rejet si couverture arrière < ce seuil
    normal_support_angle_deg: float = 0.0  # opt-in: couverture avec normales compatibles; 0 désactive
    visibility_reverse_enabled: bool = True  # remplace le gate géométrique si le volume est assez observé
    visibility_min_known_fraction: float = 0.05  # part min du modèle classée libre/occupée
    visibility_min_known_points: int = 30  # observations min avant d'utiliser la visibilité
    min_structure_height: float = 0.15  # rejet si the points expliqués sont quasi
                                       # plats (sol) : extension verticale < ce seuil (m)
    min_thickness: float = 0.06        # rejet si la région expliquée est une feuille
                                       # mince (mur/rebord) : plus petite dispersion < seuil (m)
    model_zone_grid: int = 2           # découpage de la boîte modèle pour vérifier une preuve distribuée
    model_zone_min_support: float = 0.15  # part de points expliqués pour valider une zone
    min_model_zone_fraction: float = 0.35  # fraction minimale des zones du modèle soutenues
    min_support_extent_ratio: float = 0.20  # étendue expliquée / étendue modèle (moyenne géométrique)
    support_locality_weight: float = 0.10  # pénalité des preuves concentrées dans une petite région
    # 0.50 rejetait des poses visuellement solides juste sous la frontière
    # (H=0.475-0.484). Les garde-fous de visibilité et d'étendue restent
    # responsables des faux positifs franchement incompatibles.
    null_hypothesis_score: float = 0.46  # en dessous, « none objet » gagne la final selection
    surface_distance_weight: float = 0.05  # pénalité de tri: distance_surface / seuil
    verification_top_constellations: int = 50  # shortlist vérifiée (papier, sec. 6)
    max_registrations_per_model: int = 1   # >1 conserve plusieurs poses distinctes
    registration_min_center_distance: float = 0.35  # séparation horizontale min (m)
    registration_top_constellations: int = 6  # constellations vérifiées par modèle
    multi_registration_synsets: Tuple[str, ...] = ()  # vide = toutes catégories


@dataclass
class RansacConfig:
    """1-Point RANSAC + constellations (sec. 6)."""
    geom_inlier: float = 0.15          # t_geom : distance max entre key points (m)
    desc_inlier: float = 128.0         # t_desc en unités de cellule UDF
    refine_iterations: int = 4         # itérations de raffinement des paramètres
    top_constellations: int = 50       # taille de la shortlist
    max_seeds: int = 40                # germes RANSAC testés (priorité aux grosses primitives)
    min_correspondences: int = 3       # en-dessous, on ne lance pas le RANSAC (pré-rejet)
    min_inliers: int = 3               # trois paires bijectives et spatialement distribuées
    min_scan_inliers: int = 3          # cohérent avec the trois cellules distinctes requises
    min_scan_spread: float = 0.12      # diamètre min des inliers scan (m)
    scan_cell_size: float = 0.10       # taille des cellules de support de constellation (m)
    min_scan_cells: int = 3            # cellules scan distinctes requises par constellation
    model_cell_size: float = 0.10      # taille des cellules côté modèle (avant transformation)
    min_model_cells: int = 3           # zones modèle distinctes requises par constellation
    scale_min: float = 1.0 / 1.6       # scale finale admissible (rejet des transfos dégénérées)
    scale_max: float = 1.6


@dataclass
class ClusterConfig:
    'Database descriptor clustering (paper section 6, Table 1).'
    cluster_threshold: float = 96.0    # t_cluster (un peu plus serré que t_desc)


@dataclass
class PipelineConfig:
    sdf: SDFConfig = field(default_factory=SDFConfig)
    keypoint: KeypointConfig = field(default_factory=KeypointConfig)
    primitive: PrimitiveConfig = field(default_factory=PrimitiveConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    # Convention de repère : axe "haut" du monde (ObjectSensing = Y haut après
    # mise à niveau ; la fusion estime le plan du sol pour fixer l'up).
    up_axis: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    # Categories ShapeNet utilisées dans le papier (synset IDs WordNet) :
    #   chair 03001627, table 04379243, sofa/couch 04256520.
    shapenet_synsets: Tuple[str, ...] = ("03001627", "04379243", "04256520")


DEFAULT = PipelineConfig()
