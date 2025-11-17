import math

def rotvec_to_quaternion(rx, ry, rz, order='wxyz'):
    """
    Convert rotation vector (rx,ry,rz) in radians to quaternion.
    Returns quaternion as tuple.
    order: 'wxyz' -> (w,x,y,z) (recommended)
           'xyzw' -> (x,y,z,w) (some libraries use this)
    """
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    if theta == 0.0:
        q = (1.0, 0.0, 0.0, 0.0)
    else:
        ux, uy, uz = rx/theta, ry/theta, rz/theta
        half = 0.5 * theta
        w = math.cos(half)
        s = math.sin(half)
        x = ux * s
        y = uy * s
        z = uz * s
        q = (w, x, y, z)
    if order == 'wxyz':
        return q
    elif order == 'xyzw':
        w,x,y,z = q
        return (x,y,z,w)
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'")

def quaternion_to_rotvec(x, y, z, w):
    """
    Convert quaternion (x, y, z, w) to a rotation vector (rx, ry, rz) in radians
    for UR robots.
    """
    # Normalize just in case
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    theta = 2 * math.acos(w)
    s = math.sqrt(1 - w*w)

    if s < 1e-8:  
        # Very small angle approximation
        return (x * 2, y * 2, z * 2)

    rx = x / s * theta
    ry = y / s * theta
    rz = z / s * theta

    return (rx, ry, rz)

if __name__ == "__main__":
  rx = 0
  ry = 3.14159
  rz = 0
  q = rotvec_to_quaternion(rx, ry, rz)
  print("Quaternion:", q)
  rvec = quaternion_to_rotvec(q[1], q[2], q[3], q[0])
  print("Rotation Vector:", rvec)
