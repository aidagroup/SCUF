## SCUF

**SCUF** (Spheroid Consensus Undeterministic Fitting) is an open-source implementation of the RANSAC (Random Sample Consensus) method, designed specifically for fitting ellipsoids in point clouds. It is a powerful tool for applications requiring precise detection of ellipsoidal shapes, such as object modeling, geometric analysis, and computer vision tasks. SCUF extends the classic RANSAC approach, offering specialized functionality for ellipsoid fitting.

## Installation
Requirements: Numpy, open3d

Install with [Pypi](https://pypi.org/project/scuf/):

```sh
pip3 install scuf
```
installation with 3D visualization capability (needed for demo)
```sh
pip install my_library[visualization]
```

## Take a look: 

### Example 1 - quick usage 

``` python
import numpy as np

from scuf.ransac import RANSAC

#loading point cloud
pcd = o3d.io.read_point_cloud("data/2.ply")
points = np.asarray(pcd.points)

#using ransac class
IS = RANSAC(figure = "ellipsoid")
result = IS.fit(points)

print(result)

```

### Example 2 - with visualization

``` python
import numpy as np
import open3d as o3d

from scuf.ransac import RANSAC, plot


#loading point cloud
pcd = o3d.io.read_point_cloud("data/2.ply")
o3d.visualization.draw_geometries([pcd])

points = np.asarray(pcd.points)

#using ransac class
IS = RANSAC(figure = "ellipsoid")
result = IS.fit(points)

#drawing results as points on ellispod for visualization
predict = plot(result)


o3d.visualization.draw_geometries([pcd, predict])
```


## License
[MIT](https://github.com/aidagroup/SCUF/blob/main/LICENSE)
