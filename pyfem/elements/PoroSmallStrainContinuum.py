# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers


from .Element import Element
from pyfem.util.shapeFunctions  import getElemShapeData
from pyfem.util.kinematics      import Kinematics
from numpy import zeros, dot, outer, ones , eye, ix_, linalg, tensordot 

import sys
#-------------------------------------------------------------------------------
# TODO:
# split and remove thermal implementation
# add internal force for mass balance fluid (e.g liquid water)
# add stiffness and damping matrices from linearized internal force for mass balance fluid
#-------------------------------------------------------------------------------

class ThermoSmallStrainContinuum( Element ):
  
  def __init__ ( self, elnodes , props ):
    Element.__init__( self, elnodes , props )

    self.rank = props.rank

    if self.rank == 2:
      self.dofTypes = [ 'u' , 'v' , 'p_fluid' ]
      self.nstr = 3
    elif self.rank == 3:
      self.dofTypes = [ 'u' , 'v' , 'w' , 'p_fluid' ]
      self.nstr = 6

    self.kin = Kinematics(self.rank,self.nstr)
    
    self.D     = self.material.heatConductivity*eye(self.rank)
    self.capac = self.material.heatCapacity
    
    self.alpha = self.material.alpha*ones(self.nstr)
    
    if self.rank == 2:
      self.alpha[2] = 0.;
      self.labels = [ "qx" , "qy" ]
    elif self.rank == 3:
      self.alpha[3:] = 0.;
      self.labels = [ "qx" , "qy" , "qz" ]    

    
    self.transient = True
    self.theta = 1.0

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------

  def getTangentStiffness ( self, elemdat ):
       
    sData = getElemShapeData( elemdat.coords )
    
    dDofs,pfDofs = self.splitDofIDs( len(elemdat.coords) )
    
    temp0 = elemdat.state [pfDofs] - elemdat.Dstate[pfDofs]
    
    if self.transient:
      ctt      = zeros(shape=(len(pfDofs),len(pfDofs)))
      invdtime = 1.0/self.solverStat.dtime
                       
    for iInt,iData in enumerate(sData):
      
      B = self.getBmatrix( iData.dhdx )

      self.kin.strain  = dot ( B , elemdat.state [dDofs] )
      self.kin.dstrain = dot ( B , elemdat.Dstate[dDofs] )
      
      temp     = sum( iData.h * elemdat.state [pfDofs] )
      dtemp    = sum( iData.h * elemdat.Dstate[pfDofs] )
      gradTemp = dot( iData.dhdx.transpose() , elemdat.state [pfDofs] )
            
      self.kin.strain[:self.nstr]  += -self.alpha * temp
      self.kin.dstrain[:self.nstr] += -self.alpha * dtemp 
            
      sigma,tang = self.mat.getStress( self.kin )
      
      elemdat.stiff[ix_(dDofs,dDofs)] += \
        dot ( B.transpose() , dot ( tang , B ) ) * iData.weight
        
      dsdt = -1.0 * dot( tang , self.alpha )  
      elemdat.stiff[ix_(dDofs,pfDofs)] += \
        dot ( B.transpose() , outer ( dsdt , iData.h ) ) * iData.weight
      
      elemdat.stiff[ix_(pfDofs,pfDofs)] += \
        dot ( iData.dhdx , dot( self.D , iData.dhdx.transpose() ) ) * iData.weight
  
      elemdat.fint[dDofs] += dot ( B.transpose() , sigma ) * iData.weight
      
      if self.transient:
        ctt += self.capac * outer( iData.h , iData.h ) * iData.weight
              
      self.appendNodalOutput( self.mat.outLabels() , self.mat.outData() )
      self.appendNodalOutput( self.labels , dot(self.D,gradTemp) ) 
    
    if self.transient:  
      ktt0 = invdtime * ctt - elemdat.stiff[ix_(pfDofs,pfDofs)] * \
        ( 1.0-self.theta )
      
      elemdat.stiff *= self.theta
      
      elemdat.stiff[ix_(pfDofs,pfDofs)] += invdtime * ctt 
        
    elemdat.fint[pfDofs] += \
      dot ( elemdat.stiff[ix_(pfDofs,pfDofs)] , elemdat.state[pfDofs] )
      
    if self.transient:
      elemdat.fint[pfDofs] += -dot ( ktt0 , temp0 )
     
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------

  def getInternalForce ( self, elemdat ):
     
    sData = getElemShapeData( elemdat.coords )
    
    dDofs,pfDofs = self.splitDofIDs( len(elemdat.coords) )
    
    temp0 = elemdat.state [pfDofs] - elemdat.Dstate[pfDofs]
    
    stiff = zeros(shape=(4,4))
    
    if self.transient:
      ctt = zeros(shape=(4,4))
      invdtime = 1.0/self.solverStat.dtime
                 
    for iInt,iData in enumerate(sData):
      
      B = self.getBmatrix( iData.dhdx )

      self.kin.strain  = dot ( B , elemdat.state [dDofs] )
      self.kin.dstrain = dot ( B , elemdat.Dstate[dDofs] )
      
      temp     = sum( iData.h * elemdat.state [pfDofs] )
      dtemp    = sum( iData.h * elemdat.Dstate[pfDofs] )
      gradTemp = dot( iData.dhdx.transpose() , elemdat.state [pfDofs] )
            
      self.kin.strain[:self.nstr]  += -self.alpha * temp
      self.kin.dstrain[:self.nstr] += -self.alpha * dtemp 
            
      sigma,tang = self.mat.getStress( self.kin )
            
      stiff[ix_(pfDofs,pfDofs)] += \
        dot ( iData.dhdx , dot( self.D , iData.dhdx.transpose() ) ) * iData.weight
  
      elemdat.fint[dDofs] += dot ( B.transpose() , sigma ) * iData.weight
      
      if self.transient:
        ctt += self.capac * outer( iData.h , iData.h ) * iData.weight
              
      self.appendNodalOutput( self.mat.outLabels() , self.mat.outData() )
      self.appendNodalOutput( self.labels , dot(self.D,gradTemp) ) 
    
    if self.transient:  
      stiff *= self.theta
      
      stiff += invdtime * ctt 
        
    elemdat.fint[pfDofs] += dot ( stiff , elemdat.state[pfDofs] )
      
    if self.transient:
      elemdat.fint[pfDofs] += -dot ( ktt0 , temp0 )
       
#-------------------------------------------------------------------------------
#  getBmatrix
#-------------------------------------------------------------------------------

  def getBmatrix( self , dphi ):
       
    b = zeros( shape=( self.nstr , self.rank*len(dphi) ) )

    if self.rank == 2:
      for i,dp in enumerate(dphi):
        b[0,i*2+0] = dp[0]
        b[1,i*2+1] = dp[1]
        b[2,i*2+0] = dp[1]
        b[2,i*2+1] = dp[0]
    elif self.rank == 3:
      for i,dp in enumerate(dphi):
        b[0,i*3+0] = dp[0]
        b[1,i*3+1] = dp[1]
        b[2,i*3+2] = dp[2]

        b[3,i*3+1] = dp[2]
        b[3,i*3+2] = dp[1]

        b[4,i*3+0] = dp[2]
        b[4,i*3+2] = dp[0]

        b[5,i*3+0] = dp[1]
        b[5,i*3+1] = dp[0]
   
    return b

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------

  def splitDofIDs( self , n ):
  
    '''Routine to split the dof IDs in two groups, one for the displacement 
       degrees of freedom, the second for the pore fluid degrees of freedom. 
       n is the number of degrees of freedom in this model'''
    
    if self.rank == 2:
      if n == 3:
        return [0,1,3,4,6,7],[2,5,8]
      elif n == 4:
        return [0,1,3,4,6,7,9,10],[2,5,8,11]
    elif self.rank == 3:
      if n == 4:
        return [0,1,2,4,5,6,8,9,10,12,13,14],[3,7,11,15]
      elif n == 6:
        return [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22],[3,7,11,15,19,23]
      elif n == 8:
        return [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30],[3,7,11,15,19,23,27,31]
    else:
      print("Error")
