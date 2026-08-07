<h1>LISATracks</h1>

Package to make animations of gravitational wave strain as function of time, for sources in LISA. Built on top of the amazing <a href="https://www.manim.community/">Manim</a> animation library. Currently under development.  

https://github.com/user-attachments/assets/12b2da8c-53eb-4a4d-9ea6-5251fc9bc6e7

Currently supported sources: 
- 
<ul>
  <li>Stellar origin binary inspiral, modeled using a custom implementation of <a href="https://arxiv.org/abs/1605.00304">TaylorF2-Eccentric waveform</a>.</li>
  <li>Massive black hole binary inspiral, modelled via <a href="https://github.com/asantini29/phentax">PhenTAX</a>.</li>
  <li>Monochromatic Galactic binaries, modelled using simple 0 PN constant amplitude.</li>
  
</ul>
EMRIs will be added in the near future...

PSD contains time dependent confusion noise (using implementation from Balrog).

Light mode animation option also available:

https://github.com/user-attachments/assets/afe41ac0-4652-4b02-a6ff-46eacd49bc5a

The above dark and light mode animations can be made from the following <a href="https://github.com/dig07/LISATracks/blob/main/Examples/Multiple_sources_animation.ipynb">Notebook</a>.
