from __future__ import print_function
import os
import re
import xml.etree.ElementTree as ET


def read_vivado_report(hls_dir, full_report=False):
    if not os.path.exists(hls_dir):
        print('Path {} does not exist. Exiting.'.format(hls_dir))
        return

    prj_dir = None
    top_func_name = None

    if os.path.isfile(hls_dir + '/build_prj.tcl'):
        prj_dir, top_func_name = _parse_build_script(hls_dir + '/build_prj.tcl')

    if prj_dir is None or top_func_name is None:
        print('Unable to read project data. Exiting.')
        return

    sln_dir = hls_dir + '/' + prj_dir
    if not os.path.exists(sln_dir):
        print('Project {} does not exist. Rerun "hls4ml build -p {}".'.format(prj_dir, hls_dir))
        return

    solutions = _find_solutions(sln_dir)
    print('Found {} solution(s) in {}.'.format(len(solutions), sln_dir))

    for sln in solutions:
        print('Reports for solution "{}":\n'.format(sln))
        _find_reports(sln_dir + '/' + sln, top_func_name, full_report)


def _parse_build_script(script_path):
    prj_dir = None
    top_func_name = None

    with open(script_path, 'r') as f:
        for line in f.readlines():
            if 'open_project' in line:
                prj_dir = line.split()[-1]
            elif 'set_top' in line:
                top_func_name = line.split()[-1]

    return prj_dir, top_func_name


def _find_solutions(sln_dir):
    solutions = []

    if os.path.isfile(sln_dir + '/vivado_hls.app'):
        with open(sln_dir + '/vivado_hls.app') as f:
            # Get rid of namespaces (workaround to support two types of vivado_hls.app files)
            xmlstring = re.sub(' xmlns="[^"]+"', '', f.read(), count=1)

        root = ET.fromstring(xmlstring)
        for sln_tag in root.findall('solutions/solution'):
            sln_name = sln_tag.get('name')
            if sln_name is not None and os.path.isdir(sln_dir + '/' + sln_name):
                solutions.append(sln_name)

    return solutions


def _find_reports(sln_dir, top_func_name, full_report=False):
    csim_file = sln_dir + '/csim/report/{}_csim.log'.format(top_func_name)
    if os.path.isfile(csim_file):
        _show_csim_report(csim_file)
    else:
        print('C simulation report not found.')

    syn_file = sln_dir + '/syn/report/{}_csynth.rpt'.format(top_func_name)
    if os.path.isfile(syn_file):
        _show_synth_report(syn_file, full_report)
    else:
        print('Synthesis report not found.')

    cosim_file = sln_dir + '/sim/report/{}_cosim.rpt'.format(top_func_name)
    if os.path.isfile(cosim_file):
        _show_cosim_report(cosim_file)
    else:
        print('Co-simulation report not found.')


def _show_csim_report(csim_file):
    with open(csim_file, 'r') as f:
        print('C SIMULATION RESULT:')
        print(f.read())


def _show_synth_report(synth_file, full_report=False):
    with open(synth_file, 'r') as f:
        print('SYNTHESIS REPORT:')
        for line in f.readlines()[2:]:
            if not full_report and '* DSP48' in line:
                break
            print(line, end='')


def _show_cosim_report(cosim_file):
    with open(cosim_file, 'r') as f:
        print('CO-SIMULATION RESULT:')
        print(f.read())


def parse_vivado_report(hls_dir):
    if not os.path.exists(hls_dir):
        print('Path {} does not exist. Exiting.'.format(hls_dir))
        return

    prj_dir = None
    top_func_name = None

    if os.path.isfile(hls_dir + '/build_prj.tcl'):
        prj_dir, top_func_name = _parse_build_script(hls_dir + '/build_prj.tcl')

    if prj_dir is None or top_func_name is None:
        print('Unable to read project data. Exiting.')
        return

    sln_dir = hls_dir + '/' + prj_dir
    if not os.path.exists(sln_dir):
        print('Project {} does not exist. Rerun "hls4ml build -p {}".'.format(prj_dir, hls_dir))
        return

    solutions = _find_solutions(sln_dir)
    if len(solutions) > 1:
        print('WARNING: Found {} solution(s) in {}. Using the first solution.'.format(len(solutions), sln_dir))

    report = {}

    sim_file = hls_dir + '/tb_data/csim_results.log'
    if os.path.isfile(sim_file):
        csim_results = []
        with open(sim_file, 'r') as f:
            for line in f.readlines():
                csim_results.append([float(r) for r in line.split()])
        report['CSimResults'] = csim_results

    sim_file = hls_dir + '/tb_data/rtl_cosim_results.log'
    if os.path.isfile(sim_file):
        cosim_results = []
        with open(sim_file, 'r') as f:
            for line in f.readlines():
                cosim_results.append([float(r) for r in line.split()])
        report['CosimResults'] = cosim_results

    syn_file = sln_dir + '/' + solutions[0] + '/syn/report/{}_csynth.xml'.format(top_func_name)
    if os.path.isfile(syn_file):
        root = ET.parse(syn_file).getroot()

        # Performance
        perf_node = root.find('./PerformanceEstimates')
        report['EstimatedClockPeriod'] = perf_node.find('./SummaryOfTimingAnalysis/EstimatedClockPeriod').text
        report['BestLatency'] = perf_node.find('./SummaryOfOverallLatency/Best-caseLatency').text
        report['WorstLatency'] = perf_node.find('./SummaryOfOverallLatency/Worst-caseLatency').text
        report['IntervalMin'] = perf_node.find('./SummaryOfOverallLatency/Interval-min').text
        report['IntervalMax'] = perf_node.find('./SummaryOfOverallLatency/Interval-max').text
        # Area
        area_node = root.find('./AreaEstimates')
        for child in area_node.find('./Resources'):
            report[child.tag] = child.text
        for child in area_node.find('./AvailableResources'):
            report['Available' + child.tag] = child.text
    else:
        print('Synthesis report not found.')

    cosim_file = sln_dir + '/' + solutions[0] + '/sim/report/{}_cosim.rpt'.format(top_func_name)
    if os.path.isfile(cosim_file):
        with open(cosim_file, 'r') as f:
            for line in f.readlines():
                if re.search('VHDL', line) or re.search('Verilog', line):
                    result = line[1:].split()  # [1:] skips the leading '|'
                    result = [res[:-1] if res[-1] == '|' else res for res in result]
                    # RTL, Status, Latency-min, Latency-avg, Latency-max, Interval-min, Interval-avg, Interval-max
                    if result[1] == 'NA':
                        continue
                    else:
                        report['CosimRTL'] = result[0]
                        report['CosimStatus'] = result[1]
                        report['CosimLatencyMin'] = result[2]
                        report['CosimLatencyMax'] = result[4]
                        report['CosimIntervalMin'] = result[5]
                        report['CosimIntervalMax'] = result[7]

    return report
