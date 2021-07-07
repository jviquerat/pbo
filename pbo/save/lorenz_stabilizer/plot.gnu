### Retrieve arguments
path = ARG1
file = ARG2

### Settings
reset
set print "-"
set term png truecolor size 1500,500
set output path."/lorenz.png"
set grid
set style fill transparent solid 0.25 noborder
set style line 1  lt 1  lw 2 pt 3 ps 0.5
set style line 2  lt 2  lw 2 pt 3 ps 0.5
set style line 3  lt 3  lw 2 pt 3 ps 0.5

### Global png
set multiplot layout 3,1
file = path.file
set arrow from 0, graph 0 to 0, graph 1 nohead lt rgb "black" lw 2

# Plot x, y, z
plot file u 1:2 w l ls 1 t "x"
plot file u 1:3 w l ls 2 t "y"
plot file u 1:4 w l ls 3 t "z"

### Animated gif
unset multiplot
reset
set term gif animate delay 5 size 1200,600
set output path."/lorenz.gif"
unset colorbox
unset border
unset key
unset tics
set size ratio -1
set xrange [-30:30]
set yrange [-30:30]
set zrange [0:60]


n_steps = system(sprintf('cat %s | wc -l', file))
n_imgs  = 500
steps   = floor(n_steps/n_imgs)

# Plot all points sampling
do for [i=0:n_imgs] {
   splot file u 2:3:4:2 every ::0::i*steps w l palette lw 1.5 notitle
}