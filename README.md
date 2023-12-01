# 2D FET charge density

This is a custom class that implements two functionalities:
  1. It incorporates electrostatic equations combining the geometrical capacitance of a classical bilayer semiconductor field-effect-structure with a non-linear quantum capacitance coming from the electronic density of states (DOS) inside
     the heterostructure. The final charge density is computer by the scipy root method. 
     
![dualgate_device](https://github.com/borya-polovnikov/2D_FET_charge_density/assets/147932035/730994a3-40b0-48cf-8bf4-b45311321bf7)

  2. It provides the class LineDrawer to pass arbitrary functional forms of the structure's quantum capacitance / DOS by drawing arbitrary forms of $n(E) = \int_0^E DOS(E') dE' $, see the video of the interface below:

     

https://github.com/borya-polovnikov/2D_FET_charge_density/assets/147932035/638e74bc-0b94-4ee7-8dc7-0786e415ba74

In the functional form of $n(E)$, plateaus correspond to $DOS(E) = dn(E)/dE = 0$, whereas steps in $n$ represent local peaks in the DOS.
