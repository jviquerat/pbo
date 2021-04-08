#! /usr/bin/gnuplot

### Specific declarations
n_gen    = 100
n_pop    = 8
x_min    =-0.1
x_max    = 0.1
y_min    =-0.1
y_max    = 0.1
rad      = 0.025
file     = ARG1

# Set styles
NEW_REG_PTS = 'pointtype 7 pointsize 1.5 linecolor rgb "blue"'
REG_PTS     = 'pointtype 1 pointsize 1.5 linecolor rgb "purple"'

### Animated gif
# Initial stuff
reset
set term gif animate delay 50 size 1200,600 enhanced crop
set output "pbo.gif"
set key right bottom
set grid

# Plot all points sampling
set xrange [x_min:x_max]
set yrange [y_min:y_max]

do for [i=0:n_gen-1] {
   set multiplot layout 1,2

       # Setup
       set title "generation: ".i
       start = n_pop*i
       end   = n_pop*(i+1) - 1
       cor   = 0
       if (i = 0) {cor = n_pop}

       # Set objects
       set object 1 rect from -rad,-rad to rad,rad fc lt 2 front
       set object 2 rect from -0.045,-0.045 to 0.045,0.045 fs empty border rgb "black" lw 1
       set object 3 rect from -0.08,-0.08 to 0.08,0.08 fs empty border rgb "black" lw 1

       # Plot gif
       plot file every ::0::end-cor u 4:5 w p t "prev gen" @REG_PTS, \
            file every ::start::end u 4:5 w p t "new  gen" @NEW_REG_PTS

       # Plot png
       set title "Every point"
       plot file u 4:5 title "all individuals" w p @REG_PTS

   unset multiplot
}