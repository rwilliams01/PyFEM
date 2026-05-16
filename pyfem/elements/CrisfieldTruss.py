# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

from .Element import Element
from pyfem.util.transformations import toElementCoordinates, toGlobalCoordinates

from numpy import zeros, dot, array, eye, outer
from scipy.linalg import norm

# TOOD: implement the 2D element first
# TODO: we retirve the coordinates
# TODO: implement internla force 2d and 3d, the same for he siffntes with rank we should by pass it
class Truss ( Element ):

  #Number of dofs per element
  dofTypes = ['u','v']
  
  def __init__ ( self, elnodes , props ):
    Element.__init__( self, elnodes , props )

    #Initialize the history parameter
    self.setHistoryParameter( 'sigma', 0. )
    self.commitHistory()
    
    self.family = "BEAM"

#    self.rank = props.rank

#    if self.rank == 2:
#      self.dofTypes = [ 'u' , 'v' ]
#      self.nstr = 3
#      #self.outputLabels = ["s11","s22","s12"]
#    elif self.rank == 3:
#      self.dofTypes = [ 'u' , 'v' , 'w' ]
#      self.nstr = 6

  def getTangentStiffness ( self, elemdat ):


    # reference implementation
    #------------------------------------------
    a  = toElementCoordinates( elemdat.state , elemdat.coords )
    Da = toElementCoordinates( elemdat.Dstate, elemdat.coords )
    a0 = a - Da

    #Compute l0 and strains

    self.l0 = norm( elemdat.coords[1]-elemdat.coords[0] )
   
    ( epsilon , Depsilon ) = self.getStrain( a , a0 )

    #Compute the stress increment (multiplied
    #with the undeformed cross-sectional area)
    Dsigma = elemdat.props.E * Depsilon

    #Compute the current stress (multiplied
    #with the undeformed cross-sectional area)
    sigma = self.getHistoryParameter('sigma') + Dsigma

    #Update the history parameter
    self.setHistoryParameter( 'sigma', sigma )

    #Compute BL in the element coordinate system

    BL = self.getBL( a )

    #Compute the element stiffness in the element coordinate system
    KL = elemdat.props.E * elemdat.props.Area * self.l0 * outer( BL , BL )

    KNL = self.getKNL( sigma , elemdat.props.Area )

    elStiff = KL + KNL

    #Rotate element tangent stiffness to the global coordinate system
    elemdat.stiff = toGlobalCoordinates( elStiff , elemdat.coords )

    #Compute the element internal force vector in the element coordinate system
    elFint = self.l0 * sigma * elemdat.props.Area * BL

    #Rotate element fint to the global coordinate system
    elemdat.fint = toGlobalCoordinates( elFint , elemdat.coords )

#-----------------------------------------------------------------

  def getInternalForce ( self, elemdat ):

    # 2D:
    # coordinate differences
    a = X[e1, 0] + u[e1, 0] - X[e2, 0] - u[e2, 0]
    b = X[e1, 1] + u[e1, 1] - X[e2, 1] - u[e2, 1]

    # current length squared
    l2 = a**2 + b**2

    # Green-Lagrange stress
    S11 = (E / (2 * L**2)) * (l2 - L**2)

    fac = (S11 * A0) / L

    f_int = fac * np.array([a, b, -a, -b])

    # geometric stiffness matrix
    k_geo = (S11 * A0 / L) * np.array([
        [ 1,  0, -1,  0],
        [ 0,  1,  0, -1],
        [-1,  0,  1,  0],
        [ 0, -1,  0,  1]
    ])

    # material stiffness matrix
    k_mat = (E * A0 / L**3) * np.array([
        [ a**2,  a*b,   -a**2, -a*b],
        [ a*b,   b**2,  -a*b,  -b**2],
        [-a**2, -a*b,    a**2,  a*b],
        [-a*b,  -b**2,   a*b,   b**2]
    ])

    # total tangent stiffness
    k = k_geo + k_mat

    # 3D:
    a = X[e1, 0] + u[e1, 0] - X[e2, 0] - u[e2, 0]
    b = X[e1, 1] + u[e1, 1] - X[e2, 1] - u[e2, 1]
    c = X[e1, 2] + u[e1, 2] - X[e2, 2] - u[e2, 2]

    l2 = a*a + b*b + c*c

    S11 = (E / (2 * L**2)) * (l2 - L**2)

    fac = (S11 * A0) / L

    f_int = fac * np.array([a, b, c, -a, -b, -c])

    k_geo = (S11 * A0 / L) * np.array([
        [ 1, 0, 0, -1, 0, 0],
        [ 0, 1, 0, 0, -1, 0],
        [ 0, 0, 1, 0, 0, -1],
        [-1, 0, 0, 1, 0, 0],
        [ 0,-1, 0, 0, 1, 0],
        [ 0, 0,-1, 0, 0, 1]
    ])

    k_mat = (E * A0 / L**3) * np.array([
        [ a*a, a*b, a*c, -a*a, -a*b, -a*c],
        [ a*b, b*b, b*c, -a*b, -b*b, -b*c],
        [ a*c, b*c, c*c, -a*c, -b*c, -c*c],
        [-a*a,-a*b,-a*c,  a*a,  a*b,  a*c],
        [-a*b,-b*b,-b*c,  a*b,  b*b,  b*c],
        [-a*c,-b*c,-c*c,  a*c,  b*c,  c*c]
    ])

    k = k_geo + k_mat
    # reference implementation
    #------------------------------------------

    #Compute the current state vector

    a  = toElementCoordinates( elemdat.state , elemdat.coords )
    Da = toElementCoordinates( elemdat.Dstate, elemdat.coords )

    a0 = a - Da

    #Compute l0 and strains
    self.l0 = norm( elemdat.coords[1]-elemdat.coords[0] )

    ( epsilon , Depsilon ) = self.getStrain( a , a0 )
  
    #Compute the stress increment (multiplied
    #with the undeformed cross-sectional area)
    Dsigma = elemdat.props.E * Depsilon

    #Compute the current stress (multiplied
    #with the undeformed cross-sectional area)
    sigma = self.getHistoryParameter('sigma') + Dsigma

    #Update the history parameter
    self.setHistoryParameter( 'sigma', sigma )

    #Compute BL in the parent element coordinate system
    BL = self.getBL( a )

    #Compute the element internal force vector in the element coordinate system
    elFint = self.l0 * sigma * elemdat.props.Area * BL

    #Rotate element fint to the global coordinate system
    elemdat.fint = toGlobalCoordinates( elFint, elemdat.coords )

#------------------------------------------

  def getStrain( self , a , a0 ):

    epsilon  = (a[2]-a[0])/self.l0 + 0.5*((a[2]-a[0])/self.l0)**2 + 0.5*((a[3]-a[1])/self.l0)**2
    epsilon0 = (a0[2]-a0[0])/self.l0 + 0.5*((a0[2]-a0[0])/self.l0)**2 + 0.5*((a0[3]-a0[1])/self.l0)**2

    #Compute the strain increment
    Depsilon = epsilon -epsilon0

    return epsilon,Depsilon

#-------------------------------------------

  def getBL( self , a ):

    BL = zeros( 4 )

    BL[0] = (-1./self.l0)*(1.+(a[2]-a[0])/self.l0)
    BL[1] = (-1./self.l0)*(a[3]-a[1])/self.l0
    BL[2] = -BL[0]
    BL[3] = -BL[1]

    return BL
#-------------------------------------------

  def getKNL( self , sigma , A0 ):

    KNL = zeros( (4,4) )

    KNL[:2,:2] =  (sigma * A0/self.l0)*eye(2)
    KNL[:2,2:] = -(sigma * A0/self.l0)*eye(2)
    KNL[2:,:2] = KNL[:2,2:]
    KNL[2:,2:] = KNL[:2,:2]

    return KNL
