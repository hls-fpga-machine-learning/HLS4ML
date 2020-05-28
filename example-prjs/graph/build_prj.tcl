#################
#    HLS4ML
#################
array set opt {
  reset      0
  csim       1
  synth      1
  cosim      1
  validation 1
  export     0
  vsynth     0
}

foreach arg $::argv {
  foreach o [lsort [array names opt]] {
    regexp "$o=+(\\w+)" $arg unused opt($o)
  }
}

proc report_time { op_name time_start time_end } {
  set time_taken [expr $time_end - $time_start]
  set time_s [expr ($time_taken / 1000) % 60]
  set time_m [expr ($time_taken / (1000*60)) % 60]
  set time_h [expr ($time_taken / (1000*60*60)) % 24]
  puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}

# Compare file content: 1 = same, 0 = different
proc compare_files {file_1 file_2} {
    # Check if files exist, error otherwise
    if {! ([file exists $file_1] && [file exists $file_2])} {
        return 0
    }
    # Files with different sizes are obviously different
    if {[file size $file_1] != [file size $file_2]} {
        return 0
    }

    # String compare the content of the files
    set fh_1 [open $file_1 r]
    set fh_2 [open $file_2 r]
    set equal [string equal [read $fh_1] [read $fh_2]]
    close $fh_1
    close $fh_2
    return $equal
}

file mkdir tb_data
set CSIM_RESULTS "./tb_data/csim_results.log"
set RTL_COSIM_RESULTS "./tb_data/rtl_cosim_results.log"

if {$opt(reset)} {
  open_project -reset myproject_prj
} else {
  open_project myproject_prj
}
set_top myproject
add_files firmware/myproject.cpp -cflags "-std=c++0x -I[file normalize ../../hls4ml/templates/vivado]"
add_files -tb myproject_test.cpp -cflags "-std=c++0x -I[file normalize ../../hls4ml/templates/vivado]"
add_files -tb firmware/weights
add_files -tb tb_data
if {$opt(reset)} {
  open_solution -reset "solution1"
} else {
  open_solution "solution1"
}
catch {config_array_partition -maximum_size 4096}
set_part {xcku115-flvb2104-2-i}
create_clock -period 5 -name default


if {$opt(csim)} {
  puts "***** C SIMULATION *****"
  set time_start [clock clicks -milliseconds]
  csim_design
  set time_end [clock clicks -milliseconds]
  report_time "C SIMULATION" $time_start $time_end
}

if {$opt(synth)} {
  puts "***** C/RTL SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]
  csynth_design
  set time_end [clock clicks -milliseconds]
  report_time "C/RTL SYNTHESIS" $time_start $time_end
}

if {$opt(cosim)} {
  puts "***** C/RTL SIMULATION *****"
  # TODO: This is a workaround (Xilinx defines __RTL_SIMULATION__ only for SystemC testbenches).
  add_files -tb myproject_test.cpp -cflags "-std=c++0x -DRTL_SIM -I[file normalize ../../hls4ml/templates/vivado]"
  set time_start [clock clicks -milliseconds]
  cosim_design -trace_level all
  set time_end [clock clicks -milliseconds]
  report_time "C/RTL SIMULATION" $time_start $time_end
}

if {$opt(validation)} {
  puts "***** C/RTL VALIDATION *****"
  if {[compare_files $CSIM_RESULTS $RTL_COSIM_RESULTS]} {
      puts "INFO: Test PASSED"
  } else {
      puts "ERROR: Test failed"
      puts "ERROR: - csim log:      $CSIM_RESULTS"
      puts "ERROR: - RTL-cosim log: $RTL_COSIM_RESULTS"
      exit 1
  }
}

if {$opt(export)} {
  puts "***** EXPORT IP *****"
  set time_start [clock clicks -milliseconds]
  export_design -format ip_catalog
  set time_end [clock clicks -milliseconds]
  report_time "EXPORT IP" $time_start $time_end
}

if {$opt(vsynth)} {
  puts "***** VIVADO SYNTHESIS *****"
  if {[file exist myproject_prj/solution1/syn/vhdl]} {
    set time_start [clock clicks -milliseconds]
    exec vivado -mode batch -source vivado_synth.tcl >@ stdout
    set time_end [clock clicks -milliseconds]
    report_time "VIVADO SYNTHESIS" $time_start $time_end
  } else {
    puts "ERROR: Cannot find generated VHDL files. Did you run C synthesis?"
    exit 1
  }
}

exit
