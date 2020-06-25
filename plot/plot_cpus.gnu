### Retrieve arguments
path = ARG1

### Settings
reset
set print "-"
set term png truecolor size 600,600
set output "pbo_cpus.png"
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
cpu_1 = path."cpu_1.dat"
cpu_2 = path."cpu_2.dat"
cpu_4 = path."cpu_4.dat"
cpu_6 = path."cpu_6.dat"

# Plot reward
set logscale y
set format y "10^{%L}"
plot cpu_1 u 1:3:4 w filledc lt 1 notitle,            \
     cpu_1 u 1:2  smooth csplines w l ls 1 t "1 cpu", \
     cpu_2 u 1:3:4 w filledc lt 2 notitle,            \
     cpu_2 u 1:2  smooth csplines w l ls 2 t "2 cpu", \
     cpu_4 u 1:3:4 w filledc lt 3 notitle,            \
     cpu_4 u 1:2  smooth csplines w l ls 3 t "4 cpu", \
     cpu_6 u 1:3:4 w filledc lt 4 notitle,            \
     cpu_6 u 1:2  smooth csplines w l ls 4 t "6 cpu", \