import matplotlib.pyplot as plt
import numpy as np


from typing import Tuple, List, Callable


def solution_co_ordinates(
    point: np.ndarray, matrix: np.ndarray, time_steps: int
) -> Tuple[List[float], List[float]]:
    """Traces a solution to a system of differential equations at a given point

    Parameters
    ----------
    point : np.ndarray
        A point from which to trace the solution to the system of differential equations
    matrix : np.ndarray
        The matrix representing the system of linear differential equations
    time_steps : int
        The number of time steps to trace the solution

    Returns
    -------
    Tuple[List[float], List[float]]
        A tuple containing two lists, the x and y coordinates of the solution
    """
    x_i, y_i = point
    xs, ys = [x_i], [y_i]
    dx = 0.001

    for time in range(time_steps):
        point = np.array([x_i, y_i])
        transformed_point = matrix @ point
        x_i += transformed_point[0] * dx
        y_i += transformed_point[-1] * dx
        xs.append(x_i)
        ys.append(y_i)

    return xs, ys


def transform_meshgrid(
    X: np.ndarray, Y: np.ndarray, transformation_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms a meshgrid using a linear transformation matrix
    (to create the vector fields in the case of the system of differential equations)

    Parameters
    ----------
    X : np.ndarray
        An X grid
    Y : np.ndarray
        A Y grid

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The transformed X and Y grids
    """
    # Step 1: Stack the grid arrays to form a 2D array where each column is a point in the grid
    points = np.vstack([X.ravel(), Y.ravel()])

    # Apply the transformation matrix using matrix multiplication
    transformed_points = transformation_matrix @ points

    # Step 2: Reshape the transformed points back to the original grid shape
    X_transformed = transformed_points[0, :].reshape(X.shape)
    Y_transformed = transformed_points[1, :].reshape(Y.shape)

    return X_transformed, Y_transformed


def get_eigenvalues_eigenvectors(matrix: np.ndarray) -> np.ndarray:
    """Computes the eigenvalues of a given matrix

    Parameters
    ----------
    matrix : np.ndarray
        The matrix for which to compute the eigenvalues

    Returns
    -------
    np.ndarray
        The eigenvalues of the matrix
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors


def eigenline(x: float, eigenvector: np.ndarray) -> float:
    """Computes the y-coordinate of a point on the eigenvector line

    Parameters
    ----------
    x : float
        The x-coordinate of the point
    eigenvector : np.ndarray
        The eigenvector of the matrix

    Returns
    -------
    float
        The y-coordinate of the point on the eigenvector line
    """
    # This plots the eigenvector as a line, passing through the origin
    return eigenvector * x


def draw_eigen_lines(
    axs: plt.Axes, eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> None:
    colours = ["blue", "green"]
    for eigen_value, vector in zip(eigenvalues, eigenvectors.T):
        # We transpose the eigenvectors matrix so that each row is an eigenvector
        if np.iscomplex(eigen_value) == False:
            # Only sketch the line if the eigenvalue is real, since for imaginary eigenvalues the eigenvector is complex,
            # and thus cannot be represented in a catersian plane.
            eigen_line_co_ordinates = []
            for m in range(-200, 200):
                # Sketching of the eigen line
                eigen_line_co_ordinates.append(eigenline(m, vector))
            eigen_line_co_ordinates = np.array(eigen_line_co_ordinates)
            axs.plot(
                eigen_line_co_ordinates[:, 0],
                eigen_line_co_ordinates[:, 1],
                color=colours.pop(),
                label=f"$\lambda$ = {eigen_value:.2f}",
            )

            # We want to add arrows with the following rules:
            # If the eigenvalue is positive, we want the arrow to point towards the origin
            # If the eigenvalue is negative, we want the arrow to point away from the origin
            # This is quite a crude solution, but it works for the purposes of this demonstration
            plt.arrow(
                10 * vector[0],
                10 * vector[1],
                eigen_value * vector[0],
                eigen_value * vector[1],
                color="red",
                width=2,
            )
            plt.arrow(
                -20 * vector[0],
                -20 * vector[1],
                -eigen_value * vector[0],
                -eigen_value * vector[1],
                color="red",
                width=2,
            )


def generate_quiver_plot(
    dimension: int, matrix: np.ndarray, resolution: int = 40
) -> Tuple[plt.Figure, plt.Axes]:
    """Generates a quiver plot of a vector field

    Parameters
    ----------
    dimension : int
        The dimensions of the quiver plot (i.e x and y limits.)
    matrix : np.ndarray
        The matrix representing the vector field
    resolution : int, optional
        Number of arrows to plot , by default 40

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        A tuple containing the figure and axes of the plot
    """
    # Create a meshgrid of points
    space = np.linspace(-dimension, dimension, resolution)

    x, y = np.meshgrid(space, space)

    # Transforms the points on the grid according to the vector field
    u, v = transform_meshgrid(x, y, matrix)

    fig, axs = plt.subplots()
    axs.quiver(x, y, u, v)
    step = dimension // 10

    # We also want to plot the invariant lines traced by the eigenvectors of our system of equations on the quiver plot
    eigenvalues, eigenvectors = get_eigenvalues_eigenvectors(matrix)
    draw_eigen_lines(axs, eigenvalues, eigenvectors)

    # Add a legend to the plot
    plt.legend()

    # Add axes to the plot
    axs.axhline(y=0, color="k")
    axs.axvline(x=0, color="k")

    # add a title

    axs.set_title("Interactive Linear Phase Portrait")
    return fig, axs


def onclick(event, matrix: np.ndarray, dimension: int) -> None:
    """Event handler for the button press event

    parameters:
    -----------
    event: Event
        The event object
    matrix: np.ndarray
        The matrix representing the system of linear differential equations
    dimension: int
        The dimensions of the quiver plot (i.e x and y limits.)

    returns:
    --------
    None
    """
    if ax := event.inaxes:
        # Extract x and y co-ordinates of the point clicked
        x, y = event.xdata, event.ydata

        xs, ys = solution_co_ordinates(
            np.array([x, y]),
            matrix=matrix,
            time_steps=10_000,
        )
        ax.plot(xs, ys, color="red")
        ax.set_xlim(-dimension, dimension)
        ax.set_ylim(-dimension, dimension)
        # Draws the line onto the canvas
        fig.canvas.draw()


dimension = 100  # 100 by 100 grid

# Define the system of equations for a circle:
# dx/dt = 0x + y
# dy/dt = -x + 0y

# matrix = np.array([[0, 1], [-1, 0]])

# Define the system of equations for a saddle:
# dx/dt = 0x + y
# dy/dt = x + 0y

# matrix = np.array([[0, 1], [1, 0]])

# Define the system of equations for a stable node:
# dx/dt = -x + y
# dy/dt = -x - y

# matrix = np.array([[-1, 1], [-1, -1]])

# Define the system of equations for straight line solutions:
# dx/dt = x + y
# dy/dt = x + y

# matrix = np.array([[1, 1], [1, 1]])

# Define the figure and axes for the plot (this is a quiver plot vector field)
fig, axs = generate_quiver_plot(dimension, matrix)

# Connect the figure to the event handler.
cid = fig.canvas.mpl_connect(
    "button_press_event", lambda event: onclick(event, matrix, dimension)
)

# Show the interactive plot
plt.show()
