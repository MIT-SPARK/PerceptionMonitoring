from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List
import msgpack
import numpy as np
from typing import Optional

@dataclass
class Pose:
    timestamp: Optional[float]
    pose: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    acceleration: np.ndarray

    def __init__(
        self,
        pose=np.eye(4),
        velocity=np.zeros((3,)),
        angular_velocity=np.zeros((3,)),
        acceleration=np.zeros((3,)),
        timestamp=None,
    ):
        self.pose = pose
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.acceleration = acceleration
        self.timestamp = timestamp

    def position(self):
        return self.pose[:3, 3]

    def rotation(self):
        return self.pose[:3, :3]

@dataclass_json
@dataclass(frozen=False)
class Bag:
    mapfile:str = ""
    pose: List[Pose] = field(default_factory=list)

    def save(self, filename: str):
        enc = msgpack.packb(self.to_dict(), use_bin_type=True)
        with open(filename, "wb") as outfile:
            outfile.write(enc)

    @classmethod
    def load(cls, filename: str) -> "Bag":
        bag = cls()
        with open(filename, "rb") as data_file:
            enc = data_file.read()
        dec = msgpack.unpackb(enc, raw=False)
        for pose in dec["pose"]:
            bag.pose.append(
                Pose(
                    pose=np.array(pose["pose"]),
                    velocity=np.array(pose["velocity"]),
                    angular_velocity=np.array(pose["angular_velocity"]),
                    acceleration=np.array(pose["acceleration"]),
                    timestamp=pose["timestamp"],
                )
            )
        bag.mapfile = dec["mapfile"]
        return bag

        # print(dec)
        # return cls.from_dict(dec)
