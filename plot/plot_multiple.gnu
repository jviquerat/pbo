### Retrieve arguments
path = ARG1

### Settings
reset
set print "-"
set term png truecolor size 600,600
set output "pbo_multiple.png"
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
es = path."es.dat"
diag = path."diag.dat"
full = path."full.dat"

# Plot reward
set logscale y
set format y "10^{%L}"
plot es u 1:3:4 w filledc lt 1 notitle,                 \
     es u 1:2  smooth csplines w l ls 1 t "es",         \
     diag u 1:3:4 w filledc lt 2 notitle,               \
     diag u 1:2  smooth csplines w l ls 2 t "cma-diag", \
     full u 1:3:4 w filledc lt 3 notitle,               \
     full u 1:2  smooth csplines w l ls 3 t "cma-full", \