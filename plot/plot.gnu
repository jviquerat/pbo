### Retrieve arguments
path = ARG1

### Settings
reset
set print "-"
set term png truecolor size 500,500
set output "pbo_avg.png"
set grid
set style fill transparent solid 0.25 noborder
set style line 1  lt 1  lw 3 pt 3 ps 0.5
set style line 2  lt 2  lw 3 pt 3 ps 0.5
set style line 3  lt 3  lw 3 pt 3 ps 0.5
set style line 4  lt 4  lw 3 pt 3 ps 0.5
set style line 5  lt 5  lw 3 pt 3 ps 0.5
set style line 6  lt 6  lw 3 pt 3 ps 0.5
set style line 7  lt 7  lw 3 pt 3 ps 0.5
set style line 8  lt 8  lw 3 pt 3 ps 0.5
set style line 9  lt 9  lw 3 pt 3 ps 0.5
set style line 10 lt 10 lw 3 pt 3 ps 0.5

### Global png
#set multiplot layout 1,3
file = path."/pbo_avg.dat"

# Plot reward
set logscale y
set format y "10^{%L}"
plot file u 1:3:4 w filledc lt 1 notitle, \
     file u 1:2  smooth csplines w l ls 1 t "reward"

# Reset formats for the remaining plots
#set format y     
#unset logscale y

# Plot mu and sigma losses
#plot file u 1:6:7 w filledc lt 2 notitle, \
#     file u 1:5  smooth csplines w l ls 2 t "mu loss", \
#     file u 1:9:10 w filledc lt 3 notitle, \
#     file u 1:8  smooth csplines w l ls 3 t "sg loss"
  
# Plot mu and sigma gradient norms
#plot file u 1:12:13 w filledc lt 4 notitle, \
#     file u 1:11  smooth csplines w l ls 4 t "mu grad norm", \
#     file u 1:15:16 w filledc lt 5 notitle, \
#     file u 1:14  smooth csplines w l ls 5 t "sg grad norm"