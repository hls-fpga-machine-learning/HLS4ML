#################
#    HLS4ML
#################
array set opt {
  csim   1
  synth  1
  cosim  1
  export 1
}

foreach arg $::argv {
  foreach o [lsort [array names opt]] {
    regexp "$o +(\\w+)" $arg unused opt($o)
  }
}

open_project -reset myproject_prj
set_top myproject
add_files firmware/myproject.cpp -cflags "-I[file normalize ../../nnet_utils] -std=c++0x"
add_files -tb myproject_test.cpp -cflags "-I[file normalize ../../nnet_utils] -std=c++0x"
add_files -tb firmware/weights
#add_files -tb tb_data
open_solution -reset "solution1"
catch {config_array_partition -maximum_size 4096}
set_part {xcku115-flvb2104-2-i}
create_clock -period 5 -name default

if {$opt(csim)} {
  puts "***** C SIMULATION *****"
  csim_design
}

if {$opt(synth)} {
  puts "***** C/RTL SYNTHESIS *****"
  csynth_design
  if {$opt(cosim)} {
    puts "***** C/RTL SIMULATION *****"
    cosim_design -trace_level all
  }
  if {$opt(export)} {
    puts "***** EXPORT IP *****"
    export_design -format ip_catalog
  }
}

exit
