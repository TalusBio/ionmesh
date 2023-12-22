from matplotlib import pyplot as plt
import json
from dataclasses import dataclass


@dataclass
class BoundingBox:
    x_center: int
    y_center: int
    w: int
    h: int


@dataclass
class Point:
    x: int
    y: int


@dataclass
class QuadRepresentation:
    bounding_boxes: list[BoundingBox]
    points: list[Point]

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        for bb in self.bounding_boxes:
            bot_left = (bb.x_center - (bb.w / 2), bb.y_center - (bb.h / 2))
            ax.add_patch(
                plt.Rectangle(
                    xy=bot_left,
                    width=bb.w,
                    height=bb.h,
                    alpha=0.05,
                    facecolor="gray",
                    edgecolor="black",
                    linewidth=1,
                )
            )

        for point in self.points:
            ax.scatter(point.x, point.y, c="red", s=10)

        ax.grid(False)


def read_quad_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def parse_quad_dict(quad_dict):
    bounding_boxes = []
    points = []

    bounding_boxes.append(
        BoundingBox(
            x_center=quad_dict["boundary"]["x_center"],
            y_center=quad_dict["boundary"]["y_center"],
            w=quad_dict["boundary"]["width"],
            h=quad_dict["boundary"]["height"],
        )
    )

    if "points" in quad_dict:
        for point in quad_dict["points"]:
            points.append(Point(x=point["x"], y=point["y"]))

    for cardinal in ["northeast", "northwest", "southeast", "southwest"]:
        if cardinal in quad_dict and quad_dict[cardinal] is not None:
            northeast = parse_quad_dict(quad_dict[cardinal])
            bounding_boxes.extend(northeast.bounding_boxes)
            points.extend(northeast.points)

    return QuadRepresentation(bounding_boxes=bounding_boxes, points=points)


SAMPLE_JSON_STR = """
{
    "boundary": {
        "x": 0.0,
        "y": 0.0,
        "width": 50.0,
        "height": 50.0
    },
    "capacity": 4,
    "radius": 15,
    "points": [
        {
            "x": 20.0,
            "y": 20.0
        },
        {
            "x": 5.45,
            "y": 4.29
        },
        {
            "x": 2.69,
            "y": 9.25
        },
        {
            "x": 12.94,
            "y": 18.66
        }
    ],
    "northeast": {
        "boundary": {
            "x": 25.0,
            "y": -25.0,
            "width": 25.0,
            "height": 25.0
        },
        "capacity": 4,
        "radius": 15,
        "northeast": null,
        "northwest": null,
        "southeast": null,
        "southwest": null
    },
    "northwest": {
        "boundary": {
            "x": -25.0,
            "y": -25.0,
            "width": 25.0,
            "height": 25.0
        },
        "capacity": 4,
        "radius": 15,
        "northeast": null,
        "northwest": null,
        "southeast": null,
        "southwest": null
    },
    "southeast": {
        "boundary": {
            "x": 25.0,
            "y": 25.0,
            "width": 25.0,
            "height": 25.0
        },
        "capacity": 4,
        "radius": 15,
        "points": [
            {
                "x": 18.13,
                "y": 0.05
            },
            {
                "x": 2.5,
                "y": 14.66
            },
            {
                "x": 1.1,
                "y": 4.22
            }
        ],
        "northeast": null,
        "northwest": null,
        "southeast": null,
        "southwest": null
    },
    "southwest": {
        "boundary": {
            "x": -25.0,
            "y": 25.0,
            "width": 25.0,
            "height": 25.0
        },
        "capacity": 4,
        "radius": 15,
        "northeast": null,
        "northwest": null,
        "southeast": null,
        "southwest": null
    }
}
"""


def test_quad_json():
    quad_dict = json.loads(SAMPLE_JSON_STR)
    quad = parse_quad_dict(quad_dict)
    print(quad)

    # plot to file
    fig, ax = plt.subplots()
    quad.plot(ax=ax)
    plt.savefig("quad.png")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("quad_json_path", type=str)
    args = parser.parse_args()

    quad_dict = read_quad_json(args.quad_json_path)
    quad = parse_quad_dict(quad_dict)
    print(quad.bounding_boxes)
    fig, ax = plt.subplots()
    quad.plot(ax=ax)
    plt.savefig(Path(args.quad_json_path).with_suffix(".png"))
