#! /usr/bin/gnuplot

### Specific declarations
f(x,y)   = 1.0+(x**2+y**2)/4000.0-cos(x)*cos(y/sqrt(2.0))
n_gen    = 50
n_pop    = 6
x_min    =-10
x_max    = 10
y_min    =-10
y_max    = 10
file     = ARG1

# Set styles
NEW_REG_PTS = 'pointtype 7 pointsize 1.5 linecolor rgb "yellow"'
REG_PTS     = 'pointtype 1 pointsize 1.5 linecolor rgb "black"'
MU_PTS      = 'pointtype 7 pointsize 1.5 linecolor rgb "purple"'

### Animated gif
reset
set term gif animate delay 50 size 600,600 enhanced crop
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
set cbrange[0:2]

# Plot all points sampling
set xrange [x_min:x_max]
set yrange [y_min:y_max]

do for [i=0:n_gen-1] {
       # Gif    
       set title "generation: ".i
       start = n_pop*i
       end   = n_pop*(i+1) - 1
       cor   = 0
       if (i = 0) {cor = n_pop}

       splot f(x,y), \
             file every ::0::end-cor   u 4:5:(0) w p t "prev gen" @REG_PTS, \
             file every ::start::end   u 4:5:(0) w p t "new  gen" @NEW_REG_PTS
}