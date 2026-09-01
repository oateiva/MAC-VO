from .Interface import IOptimizer
from .TwoFramePGO import TwoFrame_PGO
from .PyposeOptimizers import AnalyticModule
from .GEDF import GEDF_PGO  # registers "GEDF_PGO" in the SubclassRegistry

try:
    from .GTSAM import GTSAM_Graph  # registers "GTSAM_Graph" in the SubclassRegistry
    from .GTSAM import ISAM2_Graph  # registers "ISAM2_Graph" in the SubclassRegistry
except ImportError:
    pass  # gtsam not installed; GTSAM_Graph/ISAM2_Graph unavailable, two-frame path still works
