### Retrieve arguments
file    = ARG1
n_steps = ARG2
n_imgs  = ARG3

### Settings
reset
set print "-"
set term png truecolor size 1500,500
set output "lorenz.png"
set grid
set style fill transparent solid 0.25 noborder
set style line 1  lt 1  lw 1.5 pt 3 ps 0.5
set style line 2  lt 2  lw 1.5 pt 3 ps 0.5
set style line 3  lt 3  lw 1.5 pt 3 ps 0.5

### Global png
set multiplot layout 3,1

# Plot x, y, z
plot file u 1:2 w l ls 1 t "x"
plot file u 1:3 w l ls 2 t "y"
plot file u 1:4 w l ls 3 t "z"

### Animated gif
reset
unset multiplot
set term gif animate delay 5 size 1200,600 enhanced crop
set output "lorenz.gif"
unset grid
unset xtics
unset ytics
unset ztics
unset colorbox
set xrange [-20:20]
set yrange [-25:30]
set zrange [0:50]

steps  = floor(n_steps/n_imgs)

# Plot all points sampling
do for [i=0:n_imgs] {
       set title "step: ".i*steps
       splot file every ::0::i*steps u 2:3:4:2 w l palette lw 1.5 notitle
}