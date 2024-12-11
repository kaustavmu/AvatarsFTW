try:
    from src.handlers.input.openpose import OpenPoseFile
    from src.handlers.input.body import Body
    from src.handlers.input.silhouette import Silhouette
    from src.handlers.input.camera import Camera
    from src.handlers.input.metadata import Metadata
    from src.handlers.input.silhouette_holes import SilhouetteHoles
except (ModuleNotFoundError, ImportError) as e:
    from handlers.input.openpose import OpenPoseFile
    from handlers.input.body import Body
    from handlers.input.silhouette import Silhouette
    from handlers.input.camera import Camera
    from handlers.input.metadata import Metadata
    from handlers.input.silhouette_holes import SilhouetteHoles

__all__ = [
    'OpenPoseFile',
    'Body',
    'Silhouette',
    'Camera',
    'Metadata',
    'SilhouetteHoles',
]