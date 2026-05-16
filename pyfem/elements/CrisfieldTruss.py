# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

from .Element import Element
from pyfem.util.transformations import toElementCoordinates, toGlobalCoordinates

from numpy import zeros, dot, array, eye, outer
from scipy.linalg import norm

# coords: Nodal coordinates
# tate: Current displacement vector
# Dstate: Displacement increment
# stiff: Element stiffness matrix (output)
# fint: Internal force vector (output)

class Truss ( Element ):

  #Number of dofs per element
  dofTypes = ['u','v']
  
  def __init__ ( self, elnodes , props ):
    Element.__init__( self, elnodes , props )

    self.rank = props.rank

    if self.rank == 2:
      self.dofTypes = [ 'u' , 'v' ]
    elif self.rank == 3:
      self.dofTypes = [ 'u' , 'v' , 'w' ]

    #Initialize the history parameter
    self.setHistoryParameter( 'sigma', 0. )
    self.commitHistory()
    
    self.family = "BEAM"


  def getTangentStiffness ( self, elemdat ):

    #Compute the element tangent stiffness matrix in the global coordinate system
    if self.rank == 2:
      elemdat.stiff = self.getTangentStiffness2D (elemdat)
    elif self.rank == 3:
      elemdat.stiff = self.getTangentStiffness3D (elemdat)


#-----------------------------------------------------------------
  def getTangentStiffness2D ( self, elemdat ):

    E   = elemdat.props.E
    A0  = elemdat.props.Area
    L   = norm( elemdat.coords[1]-elemdat.coords[0] )
    # 2D:
    # coordinate differences

    X_e1 = elemdat.coords[0]
    X_e2 = elemdat.coords[1]

    u_e1 = elemdat.state[0]
    u_e2 = elemdat.state[1]

    a = X_e1[ 0] + u_e1[ 0] - X_e2[ 0] - u_e2[ 0]
    b = X_e1[ 1] + u_e1[ 1] - X_e2[ 1] - u_e2[ 1]

    # current length squared
    l2 = a**2 + b**2

    # geometric stiffness matrix
    K_geo = (S11 * A0 / L) * np.array([
        [ 1,  0, -1,  0],
        [ 0,  1,  0, -1],
        [-1,  0,  1,  0],
        [ 0, -1,  0,  1]
    ])

    # material stiffness matrix
    K_mat = (E * A0 / L**3) * np.array([
        [ a**2,  a*b,   -a**2, -a*b],
        [ a*b,   b**2,  -a*b,  -b**2],
        [-a**2, -a*b,    a**2,  a*b],
        [-a*b,  -b**2,   a*b,   b**2]
    ])

    # total tangent stiffness
    K = K_geo + K_mat

    return k
#-----------------------------------------------------------------
  def getTangentStiffness3D ( self, elemdat ):

    E   = elemdat.props.E
    A0  = elemdat.props.Area
    L = norm( elemdat.coords[1]-elemdat.coords[0] )
    # 3D:
    # coordinate differences

    X_e1 = elemdat.coords[0]
    X_e2 = elemdat.coords[1]

    u_e1 = elemdat.state[0]
    u_e2 = elemdat.state[1]

    a = X_e1[ 0] + u_e1[ 0] - X_e2[ 0] - u_e2[ 0]
    b = X_e1[ 1] + u_e1[ 1] - X_e2[ 1] - u_e2[ 1]
    c = X_e1[ 2] + u_e1[ 2] - X_e2[ 2] - u_e2[ 2]

    l2 = a*a + b*b + c*c

    # geometric stiffness matrix
    K_geo = (S11 * A0 / L) * np.array([
        [ 1, 0, 0, -1, 0, 0],
        [ 0, 1, 0, 0, -1, 0],
        [ 0, 0, 1, 0, 0, -1],
        [-1, 0, 0, 1, 0, 0],
        [ 0,-1, 0, 0, 1, 0],
        [ 0, 0,-1, 0, 0, 1]
    ])

    # material stiffness matrix
    K_mat = (E * A0 / L**3) * np.array([
        [ a*a, a*b, a*c, -a*a, -a*b, -a*c],
        [ a*b, b*b, b*c, -a*b, -b*b, -b*c],
        [ a*c, b*c, c*c, -a*c, -b*c, -c*c],
        [-a*a,-a*b,-a*c,  a*a,  a*b,  a*c],
        [-a*b,-b*b,-b*c,  a*b,  b*b,  b*c],
        [-a*c,-b*c,-c*c,  a*c,  b*c,  c*c]
    ])

    K = K_geo + K_mat

#-----------------------------------------------------------------

  def getInternalForce ( self, elemdat ):

    #Compute the element internal force vector in the global coordinate system
    if self.rank == 2:
      elemdat.fint = self.getInternalForce2D (elemdat)
    elif self.rank == 3:
      elemdat.fint = self.getInternalForce3D (elemdat)

#------------------------------------------
  def getInternalForce2D ( self, elemdat ):

    E   = elemdat.props.E
    A0  = elemdat.props.Area
    L   = norm( elemdat.coords[1]-elemdat.coords[0] )
    # 2D:
    # coordinate differences

    X_e1 = elemdat.coords[0]
    X_e2 = elemdat.coords[1]

    u_e1 = elemdat.state[0]
    u_e2 = elemdat.state[1]

    a = X_e1[ 0] + u_e1[ 0] - X_e2[ 0] - u_e2[ 0]
    b = X_e1[ 1] + u_e1[ 1] - X_e2[ 1] - u_e2[ 1]


    # current length squared
    l2 = a**2 + b**2

    # Green-Lagrange strain
    E11 = (1 / (2 * L**2)) * (l2 - L**2)
    # Second Piola-Kirchoff stres
    S11 = E*E11
#    #Update the history parameter (check acuatlly is sencond piola and not cauchy)
#    self.setHistoryParameter( 'sigma', S11 )

    fac = (S11 * A0) / L

    f_int = fac * np.array([a, b, -a, -b])


#------------------------------------------
  def getInternalForce3D ( self, elemdat ):

    E   = elemdat.props.E
    A0  = elemdat.props.Area
    L = norm( elemdat.coords[1]-elemdat.coords[0] )
    # 3D:
    # coordinate differences

    X_e1 = elemdat.coords[0]
    X_e2 = elemdat.coords[1]

    u_e1 = elemdat.state[0]
    u_e2 = elemdat.state[1]

    a = X_e1[ 0] + u_e1[ 0] - X_e2[ 0] - u_e2[ 0]
    b = X_e1[ 1] + u_e1[ 1] - X_e2[ 1] - u_e2[ 1]
    c = X_e1[ 2] + u_e1[ 2] - X_e2[ 2] - u_e2[ 2]

    l2 = a*a + b*b + c*c

    # Green-Lagrange strain
    E11 = (1 / (2 * L**2)) * (l2 - L**2)
    # Second Piola-Kirchoff stres
    S11 = E*E11
#    #Update the history parameter (check acuatlly is sencond piola and not cauchy)
#    self.setHistoryParameter( 'sigma', S11 )

    fac = (S11 * A0) / L

    f_int = fac * np.array([a, b, c, -a, -b, -c])

#------------------------------------------

  def getStrain( self , elemdat):

    L = norm( elemdat.coords[1]-elemdat.coords[0] )
    l2 = 0

    if self.rank == 2:
      X_e1 = elemdat.coords[0]
      X_e2 = elemdat.coords[1]

      u_e1 = elemdat.state[0]
      u_e2 = elemdat.state[1]

      a = X_e1[ 0] + u_e1[ 0] - X_e2[ 0] - u_e2[ 0]
      b = X_e1[ 1] + u_e1[ 1] - X_e2[ 1] - u_e2[ 1]

      # current length squared
      l2 = a**2 + b**2

    elif self.rank == 3:
      X_e1 = elemdat.coords[0]
      X_e2 = elemdat.coords[1]

      u_e1 = elemdat.state[0]
      u_e2 = elemdat.state[1]

      a = X_e1[ 0] + u_e1[ 0] - X_e2[ 0] - u_e2[ 0]
      b = X_e1[ 1] + u_e1[ 1] - X_e2[ 1] - u_e2[ 1]
      c = X_e1[ 2] + u_e1[ 2] - X_e2[ 2] - u_e2[ 2]

      l2 = a*a + b*b + c*c

    E11 = (1 / (2 * L**2)) * (l2 - L**2)

    return E11

