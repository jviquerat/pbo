#! /usr/bin/gnuplot

### Stats
method   = "ppo"
n_gen    = 30
n_pop    = 6
x_max    = 5
y_max    = 5
opt_file = "optimisation.dat"
dat_file = "database.opt.dat"

### Global png
# Initial stuff
reset
set print "-"
set term png size 1000,1000
set output method.".png"
set multiplot layout 2,2
set grid

# Set styles
NEW_REG_PTS = 'pointtype 7 pointsize 1.5 linecolor rgb "yellow"'
REG_PTS     = 'pointtype 1 pointsize 1.5 linecolor rgb "black"'

# Plot best points cost
set title "Best point cost"
plot opt_file u 1:5 title "Best point cost" @REG_PTS

# Plot best points sampling
set xrange [-x_max:x_max]
set yrange [-y_max:y_max]

set title "Best point position"
plot opt_file u 3:4 title "Best point position" @REG_PTS
unset label
unset xrange
unset yrange

# Plot all points cost
set title "Every point cost"
plot dat_file u 2:5 title "regular" @REG_PTS

# Plot all points sampling
set xrange [-x_max:x_max]
set yrange [-y_max:y_max]

set title "Every point"
plot dat_file u 3:4 title "regular" @REG_PTS

unset label
unset xrange
unset yrange

# Unset stuff
unset multiplot

### Animated gif
# Initial stuff
reset
set term gif animate delay 50 size 500,500 enhanced crop
set output method.".gif"
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
set xrange [-x_max:x_max]
set yrange [-y_max:y_max]

do for [i=1:n_gen] {
   set title method.", generation: ".i

   start = 1 + n_pop*(i-1)
   end   = n_pop*i

   splot (x-1)**2+(y-2)**2, \
         dat_file every ::1::end-n_pop+1 u 3:4:(0) w p t "prev gen" @REG_PTS, \
         dat_file every ::start::end     u 3:4:(0) w p t "new  gen" @NEW_REG_PTS
         
   
}