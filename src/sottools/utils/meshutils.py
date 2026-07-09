import matplotlib.pyplot as plt
import numpy as np

from sottools.mesh import SimplyConnectedCurrentMesh


def build_mesh_from_overlay(
    data: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_area: float = 0.1,
    refine_distance: float = 0.1,
    verbose: bool = False,
    to_center: bool = True,
):
    titles = ["Mark inlet", "Mark right boundary", "Mark outlet", "Mark left boundary"]
    plotstyles = ["ro--", "go--", "bo--", "mo--"]

    original_backend = plt.get_backend()
    plt.switch_backend("QtAgg")

    try:
        fig, ax = plt.subplots(1, 1, dpi=300)
        im = ax.imshow(
            data, origin="upper", extent=(ymin, ymax, xmax, xmin), cmap="afmhot"
        )
        plt.colorbar(im)

        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        ax.autoscale(False)

        points_inlet = []
        points_right_boundary = []
        points_outlet = []
        points_left_boundary = []

        inlet_plot = ax.plot(
            [], [], plotstyles[0], markersize=5, clip_on=False, label="Inlet", zorder=10
        )[0]
        right_boundary_plot = ax.plot(
            [],
            [],
            plotstyles[1],
            markersize=5,
            clip_on=False,
            label="Right Boundary",
            zorder=9,
        )[0]
        outlet_plot = ax.plot(
            [],
            [],
            plotstyles[2],
            markersize=5,
            clip_on=False,
            label="Outlet",
            zorder=10,
        )[0]
        left_boundary_plot = ax.plot(
            [],
            [],
            plotstyles[3],
            markersize=5,
            clip_on=False,
            label="Left Boundary",
            zorder=9,
        )[0]

        state = {"marking": "inlet", "idx": 0}
        ax.set_title(titles[state["idx"]])

        def onclick(event):
            p = event.x, event.y
            inverted_point = ax.transData.inverted().transform(p)

            if state["marking"] == "inlet":
                points_inlet.append(inverted_point)
                inlet_plot.set_data(*zip(*points_inlet, strict=True))
            elif state["marking"] == "right_boundary":
                points_right_boundary.append(inverted_point)
                points_plot = [points_inlet[-1]] + points_right_boundary
                right_boundary_plot.set_data(*zip(*points_plot, strict=True))
            elif state["marking"] == "outlet":
                points_outlet.append(inverted_point)
                points_plot_right_boundary = (
                    [points_inlet[-1]] + points_right_boundary + [points_outlet[0]]
                )
                right_boundary_plot.set_data(
                    *zip(*points_plot_right_boundary, strict=True)
                )
                outlet_plot.set_data(*zip(*points_outlet, strict=True))
            elif state["marking"] == "left_boundary":
                points_left_boundary.append(inverted_point)
                points_plot = (
                    [points_outlet[-1]] + points_left_boundary + [points_inlet[0]]
                )
                left_boundary_plot.set_data(*zip(*points_plot, strict=True))

            fig.canvas.draw_idle()

        def on_press(event):
            print("press", event.key)
            if event.key == " ":
                print("Moving on!")

                if state["marking"] == "inlet":
                    print("Finished marking inlet, now mark right boundary")
                    state["marking"] = "right_boundary"
                    state["idx"] = 1
                    ax.set_title(titles[state["idx"]])
                    fig.canvas.draw_idle()
                elif state["marking"] == "right_boundary":
                    print("Finished marking right boundary, now mark outlet")
                    state["marking"] = "outlet"
                    state["idx"] = 2
                    ax.set_title(titles[state["idx"]])
                    fig.canvas.draw_idle()
                elif state["marking"] == "outlet":
                    print("Finished marking outlet, now mark left boundary")
                    state["marking"] = "left_boundary"
                    state["idx"] = 3
                    ax.set_title(titles[state["idx"]])
                    fig.canvas.draw_idle()
                elif state["marking"] == "left_boundary":
                    print("Finished marking left boundary, done!")
                    state["marking"] = None
                    plt.close(fig)

        def on_close(event):
            state["marking"] = None
            fig.canvas.stop_event_loop()

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", on_press)
        fig.canvas.mpl_connect("close_event", on_close)
        fig.show()
        fig.canvas.start_event_loop()
    finally:
        plt.close("all")
        plt.switch_backend(original_backend)

    # Validate before proceeding
    if not points_inlet or not points_outlet:
        raise RuntimeError("Boundary marking was incomplete (window closed early?).")

    # Center the mesh points
    xmax = max(
        x
        for x, y in points_inlet
        + points_right_boundary
        + points_outlet
        + points_left_boundary
    )
    xmin = min(
        x
        for x, y in points_inlet
        + points_right_boundary
        + points_outlet
        + points_left_boundary
    )
    ymax = max(
        y
        for x, y in points_inlet
        + points_right_boundary
        + points_outlet
        + points_left_boundary
    )
    ymin = min(
        y
        for x, y in points_inlet
        + points_right_boundary
        + points_outlet
        + points_left_boundary
    )
    xcenter = (xmax + xmin) / 2
    ycenter = (ymax + ymin) / 2

    if to_center:
        points_inlet = [(x - xcenter, y - ycenter) for x, y in points_inlet]
        points_right_boundary = [
            (x - xcenter, y - ycenter) for x, y in points_right_boundary
        ]
        points_outlet = [(x - xcenter, y - ycenter) for x, y in points_outlet]
        points_left_boundary = [
            (x - xcenter, y - ycenter) for x, y in points_left_boundary
        ]

    # Now we mesh using the points collected
    points = points_inlet + points_right_boundary + points_outlet + points_left_boundary
    # Due to matplotlib plotting order, have to exchange x and y again...
    points = [(y, x) for x, y in points]
    num_segments = (
        len(points_inlet)
        + len(points_right_boundary)
        + len(points_outlet)
        + len(points_left_boundary)
    )
    segments = [[i, i + 1] for i in range(num_segments - 1)] + [
        [num_segments - 1, 0]
    ]  # Connect last point to first
    segment_markers = (
        (len(points_inlet) - 1) * [SimplyConnectedCurrentMesh.NEUMANN_IN]
        + (len(points_right_boundary) + 1)
        * [SimplyConnectedCurrentMesh.DIRICHLET_RIGHT]
        + (len(points_outlet) - 1) * [SimplyConnectedCurrentMesh.NEUMANN_OUT]
        + (len(points_left_boundary) + 1) * [SimplyConnectedCurrentMesh.DIRICHLET_LEFT]
    )
    mesh = SimplyConnectedCurrentMesh(
        points,
        segments,
        segment_markers,
        max_area=max_area,
        verbose=verbose,
        refine_distance=refine_distance,
    )

    return mesh, (xcenter, ycenter)


def draw_outline(
    data: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[np.ndarray, np.ndarray]:
    original_backend = plt.get_backend()
    plt.switch_backend("QtAgg")

    try:
        fig, ax = plt.subplots(1, 1, dpi=300)
        im = ax.imshow(
            data, origin="upper", extent=(ymin, ymax, xmax, xmin), cmap="afmhot"
        )
        plt.colorbar(im)

        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        ax.autoscale(False)

        points = []
        outline_plot = ax.plot(
            [], [], "ro--", markersize=5, clip_on=False, label="Outline", zorder=10
        )[0]

        def onclick(event):
            p = event.x, event.y
            inverted_point = ax.transData.inverted().transform(p)
            points.append(inverted_point)
            to_plot = points + [points[0]]  # Close the loop
            outline_plot.set_data(*zip(*to_plot, strict=True))
            fig.canvas.draw_idle()

        def on_close(event):
            fig.canvas.stop_event_loop()

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("close_event", on_close)
        fig.show()
        fig.canvas.start_event_loop()
    finally:
        plt.close("all")
        plt.switch_backend(original_backend)

    if not points:
        raise RuntimeError("No points were selected (window closed early?).")

    # Due to matplotlib plotting order, have to exchange x and y again...
    points = [(y, x) for x, y in points]
    # Close the loop
    points = points + [points[0]]
    return np.array(points).T
