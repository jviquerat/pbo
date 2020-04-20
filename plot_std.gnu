reset
set print "-"
set term png truecolor size 900,600
set output "parabola.png"
set grid
set style fill transparent solid 0.25 noborder
set style line 1 lt 1 lw 3 pt 3 ps 0.5
set style line 2 lt 2 lw 3 pt 3 ps 0.5
set style line 3 lt 3 lw 3 pt 3 ps 0.5
set style line 4 lt 4 lw 3 pt 3 ps 0.5

# Compute ppo std bands
set table 'ppo_lower.dat'
plot 'ppo_avg_data.dat' u 1:3 smooth cspline w l
set table 'ppo_upper.dat'
plot 'ppo_avg_data.dat' u 1:4 smooth cspline w l
unset table

set logscale y

plot "< paste ppo_lower.dat ppo_upper.dat" u 1:2:5 w filledc lt 1 notitle, \
     "ppo_avg_data.dat" u 1:2 smooth csplines w l ls 1 t "ppo avg"