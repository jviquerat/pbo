#! /usr/bin/gnuplot

### Specific declarations
a        = 1.0
b        = 5.1/(4.0*pi**2)
c        = 5.0/pi
r        = 6.0
s        = 10.0
t        = 1.0/(8.0*pi)
f(x,y)   = (a*(y-b*x**2+c*x-r)**2 + s*(1.0-t)*cos(x)+s) + 0.397887
n_gen    = 150
n_pop    = 6
x_min    =-5
x_max    = 10
y_min    = 0
y_max    = 15
file     = ARG1

# Set styles
NEW_REG_PTS = 'pointtype 7 pointsize 1.5 linecolor rgb "yellow"'
REG_PTS     = 'pointtype 1 pointsize 1.5 linecolor rgb "black"'
MU_PTS      = 'pointtype 7 pointsize 1.5 linecolor rgb "purple"'

### Animated gif
reset
set term gif animate delay 50 size 1200,600 enhanced crop
set output "pbo.gif"
set pm3d map
set isosample 100,100
set key right bottom
set grid

set palette defined ( 0 '#B2182B',\
        	    	      1 '#D6604D',\
		                  2 '#F4A582',\
		                  3 '#FDDBC7',\
		                  4 '#D1E5F0',\
		                  5 '#92C5DE',\
		                  6 '#4393C3',\
		                  7 '#2166AC' )
set cbrange[0:50]

# Plot all points sampling
set xrange [x_min:x_max]
set yrange [y_min:y_max]

do for [i=0:n_gen-1] {
   set multiplot layout 1,2

       # Gif    
       set title "generation: ".i
       start = n_pop*i
       end   = n_pop*(i+1) - 1
       cor   = 0
       if (i = 0) {cor = n_pop}

       splot f(x,y), \
             file every ::0::end-cor   u 4:5:(0) w p t "prev gen" @REG_PTS, \
             file every ::start::end   u 4:5:(0) w p t "new  gen" @NEW_REG_PTS

       # Png
       set title "Every point"
       splot file u 4:5:(0) title "all individuals" w p @REG_PTS

   unset multiplot
}