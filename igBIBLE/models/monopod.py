from typing import List
from scenario import core as scenario
from gym_ignition.utils.scenario import get_unique_model_name
from gym_ignition.scenario import model_wrapper, model_with_file


class Monopod(model_wrapper.ModelWrapper,
               model_with_file.ModelWithFile):

    def __init__(self,
                 world: scenario.World,
                 position: List[float] = (0.0, 0.0, 0.0),
                 orientation: List[float] = (1.0, 0, 0, 0),
                 model_file: str = None):

        # Get a unique model name
        model_name = get_unique_model_name(world, "monopod_v1")

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get the model file (URDF or SDF) and allow to override it
        if model_file is None:
            model_file = Monopod.get_model_file()

        # Insert the model
        ok_model = world.to_gazebo().insert_model(model_file,
                                                  initial_pose,
                                                  model_name)

        if not ok_model:
            raise RuntimeError("Failed to insert model")

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        super().__init__(model=model)

    @classmethod
    def get_model_file(cls) -> str:

        import SIMP
        return SIMP.get_model_file("monopod_v1")
