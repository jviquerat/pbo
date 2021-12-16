### Retrieve arguments
path = ARG1
type = ARG2

### Settings
reset
set print "-"
set term png truecolor size 500,500
set output "pbo_avg.png"
set grid
set style fill transparent solid 0.25 noborder
set style line 1 lt 1 lw 3 pt 3 ps 0.5

### Global png
file = path."/pbo_avg.dat"

# Plot reward
if (type eq "log") {
   set logscale y
   set format y "10^{%L}"
}

plot file u 1:3:4 w filledc lt 1 notitle, file u 1:2 w l ls 1 t "reward"