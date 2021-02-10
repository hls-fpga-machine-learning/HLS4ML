from __future__ import print_function
import tarfile
import yaml
from shutil import copyfile, copytree, rmtree
import numpy as np
import os
import re
import glob
from collections import OrderedDict

from hls4ml.writer.writers import Writer
from hls4ml.model.hls_layers import XnorPrecisionType


class VivadoWriter(Writer):

    def type_definition_cpp(self, model, atype):
        type_class = atype.__class__.__name__
        if type_class == 'HLSType':
            return 'typedef {precision} {name};\n'.format(name=atype.name, precision=atype.precision)
        elif type_class == 'CompressedType':
            cpp_fmt = ('typedef struct {name} {{ '
                       '{index} row_index;'
                       '{index} col_index;'
                       '{precision} weight; }} {name};\n')
            return cpp_fmt.format(name=atype.name, index=atype.index_precision, precision=atype.precision)
        elif type_class == 'PackedType':
            n_elem_expr = '/' if atype.unpack else '*'
            return 'typedef nnet::array<{precision}, {n_elem}> {name};\n'.format(name=atype.name, precision=atype.precision, n_elem=str(atype.n_elem) + n_elem_expr + str(atype.n_pack))
        elif type_class == 'ExponentType':
            cpp_fmt = ('typedef struct {name} {{ '
                       '{sign} sign; '
                       '{precision} weight; }} {name};\n')
            return cpp_fmt.format(name=atype.name, precision=atype.precision, sign=str(XnorPrecisionType()))
        else:
            raise Exception('Unknown data type class "{}"'.format(type_class))

    def variable_definition_cpp(self, model, var, name_suffix='', as_reference=False):
        var_class = var.__class__.__name__
        if var_class == 'ArrayVariable':
            return '{type} {name}{suffix}[{shape}]'.format(type=var.type.name, name=var.cppname, suffix=name_suffix, shape=var.size_cpp())
        elif var_class == 'StreamVariable':
            if as_reference:  # Function parameter
                return 'hls::stream<{type}> &{name}{suffix}'.format(type=var.type.name, name=var.cppname, suffix=name_suffix)
            else:  # Declaration
                return 'hls::stream<{type}> {name}{suffix}("{name}")'.format(type=var.type.name, name=var.cppname, suffix=name_suffix)
        elif var_class == 'WeightVariable':
            return '{type} {name}{suffix}[{size}]'.format(type=var.type.name, name=var.cppname, suffix=name_suffix, size=var.data_length)
        elif var_class == 'InplaceVariable':
            return None
        else:
            raise Exception('Unknown variable class "{}"'.format(var_class))

    def print_array_to_cpp(self, var, odir, write_txt_file=True):
        #######################################
        # Print weight array to C++
        #######################################

        h_file = open("{}/firmware/weights/{}.h".format(odir, var.name), "w")
        if write_txt_file:
            txt_file = open("{}/firmware/weights/{}.txt".format(odir, var.name), "w")

        # meta data
        h_file.write("//Numpy array shape {}\n".format(var.shape))
        h_file.write("//Min {:.12f}\n".format(np.min(var.min)))
        h_file.write("//Max {:.12f}\n".format(np.max(var.max)))
        h_file.write("//Number of zeros {}\n".format(var.nzeros))
        h_file.write("\n")

        h_file.write("#ifndef {}_H_\n".format(var.name.upper()))
        h_file.write("#define {}_H_\n".format(var.name.upper()))
        h_file.write("\n")

        if write_txt_file:
            h_file.write("#ifndef __SYNTHESIS__\n")
            h_file.write(var.definition_cpp() + ";\n")
            h_file.write("#else\n")

        h_file.write(var.definition_cpp() + " = {")

        # fill c++ array.
        # not including internal brackets for multidimensional case
        sep = ''
        for x in var:
            h_file.write(sep + x)
            if write_txt_file:
                txt_file.write(sep + x)
            sep = ", "
        h_file.write("};\n")
        if write_txt_file:
            h_file.write("#endif\n")
            txt_file.close()
        h_file.write("\n#endif\n")
        h_file.close()

    def write_project_dir(self, model):
        if not os.path.isdir("{}/firmware/weights".format(model.config.get_output_dir())):
            os.makedirs("{}/firmware/weights".format(model.config.get_output_dir()))

    @staticmethod
    def _make_array_pragma(variable):
        """
        Layers in hls_model.py can specify output array partitioning through the `pragma` attribute.
        If `pragma` is a string: options are 'partition', 'reshape', or 'stream'.
        If `pragma` is a tuple: (mode, type, factor) where mode is 'partition' or 'reshape', type is
        'complete', 'cyclic', or 'block', and factor is an integer only used when the type is not 'complete'.
        """

        config = variable.pragma
        if isinstance(config, tuple):
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode in ['partition', 'reshape']:
            if typ == 'complete':
                template = '#pragma HLS ARRAY_{mode} variable={name} {type} dim={dim}'
            else:
                template = '#pragma HLS ARRAY_{mode} variable={name} {type} factor={factor} dim={dim}'

            return template.format(mode=mode.upper(), name=variable.name, type=typ, factor=factor, dim=0)

        elif mode == 'stream':
            return '#pragma HLS STREAM variable={name} depth={depth}'.format(name=variable.name, depth=depth)

    def write_project_cpp(self, model):
        ###################
        # myproject.cpp
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/firmware/myproject.cpp'), 'r')
        fout = open('{}/firmware/{}.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        indent = '    '

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([self.variable_definition_cpp(model, i, as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([self.variable_definition_cpp(model, o, as_reference=True) for o in model_outputs])
                insize_str = ', '.join(['unsigned short &const_size_in_{}'.format(i) for i in range(1, len(model_inputs) + 1)])
                outsize_str = ', '.join(['unsigned short &const_size_out_{}'.format(i) for i in range(1, len(model_outputs) + 1)])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + ',\n'
                newline += indent + insize_str + ',\n'
                newline += indent + outsize_str + '\n'

            elif '//hls-fpga-machine-learning insert load weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.__class__.__name__ == 'CompressedWeightVariable':
                            newline += indent + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(w.type.name, w.nonzeros, w.name, w.name)
                        elif w.__class__.__name__ == 'ExponentWeightVariable':
                            newline += indent + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(w.type.name, w.data_length, w.name, w.name)
                        else:
                            newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(w.type.name, w.data_length, w.name, w.name)

            # Add input/output type
            elif '//hls-fpga-machine-learning insert IO' in line:
                newline = line
                all_inputs = [i.cppname for i in model_inputs]
                all_outputs = [o.cppname for o in model_outputs]
                io_type = model.config.get_config_value("IOType")

                if io_type == 'io_parallel':
                    for i in model_inputs:
                        newline += indent + self._make_array_pragma(i) + '\n'
                    for o in model_outputs:
                        newline += indent + self._make_array_pragma(o) + '\n'
                    # TODO discussed adding a handle for setting the interface mode for individual input and output arrays (16.03.2020)
                    # Probably the handle doesn't need to be exposed to the user but should be just set in hls_model.py
                    newline += indent + '#pragma HLS INTERFACE ap_vld port={},{} \n'.format(','.join(all_inputs), ','.join(all_outputs))
                    if model.config.model_strategy == 'Resource':
                        newline += indent + '#pragma HLS DATAFLOW \n'
                    else:
                        newline += indent + '#pragma HLS PIPELINE \n'
                if io_type == 'io_serial' or io_type == 'io_stream':
                    newline += indent + '#pragma HLS INTERFACE axis port={},{} \n'.format(','.join(all_inputs), ','.join(all_outputs))
                    newline += indent + '#pragma HLS DATAFLOW \n'

                inval_str = '\n    '.join(['const_size_in_{} = {};'.format(i, inp.size_cpp()) for i, inp in enumerate(model_inputs, 1)])
                outval_str = '\n    '.join(['const_size_out_{} = {};'.format(i, out.size_cpp()) for i, out in enumerate(model_outputs, 1)])
                newline += '\n' + indent + inval_str
                newline += '\n' + indent + outval_str
                newline += '\n'

            elif '//hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                inputs = model.get_input_variables()
                outputs = model.get_output_variables()
                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in inputs and var not in outputs:
                            def_cpp = self.variable_definition_cpp(model, var)
                            if def_cpp is not None:
                                newline += '    ' + def_cpp + ';\n'
                                if var.pragma:
                                    newline += '    ' + self._make_array_pragma(var) + '\n'
                    func = layer.function_cpp()
                    if func:
                        if len(func) == 1:
                            newline += '    ' + func[0] + ' // ' + layer.name + '\n'
                        else:
                            newline += '// ' + layer.name + '\n'
                            for line in func:
                                newline += '    ' + line + '\n'
                        if model.config.trace_output and layer.get_attr('Trace', False):
                            newline += '#ifndef __SYNTHESIS__\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(var.type.name, var.name, layer.name, var.size_cpp())
                            newline += '#endif\n'
                        newline += '\n'

            # Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_project_header(self, model):
        #######################
        # myproject.h
        #######################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/firmware/myproject.h'), 'r')
        fout = open('{}/firmware/{}.h'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        indent = '    '

        for line in f.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'void myproject(' in line:
                newline = 'void {}(\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([self.variable_definition_cpp(model, i, as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([self.variable_definition_cpp(model, o, as_reference=True) for o in model_outputs])
                insize_str = ', '.join(['unsigned short &const_size_in_{}'.format(i) for i in range(1, len(model_inputs) + 1)])
                outsize_str = ', '.join(['unsigned short &const_size_out_{}'.format(o) for o in range(1, len(model_outputs) + 1)])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + ',\n'
                newline += indent + insize_str + ',\n'
                newline += indent + outsize_str + '\n'
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_defines(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/firmware/defines.h'), 'r')
        fout = open('{}/firmware/defines.h'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():

            # Insert numbers
            if '//hls-fpga-machine-learning insert numbers' in line:
                newline = line
                numbers = OrderedDict.fromkeys([layer.get_numbers_cpp() for layer in model.get_layers()])
                newline += ''.join(numbers)

            elif '//hls-fpga-machine-learning insert layer-precision' in line:
                newline = line
                all_precision = OrderedDict()
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    all_precision.update(layer_precision)
                for used_type in all_precision.values():
                    newline += self.type_definition_cpp(model, used_type)

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_parameters(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/firmware/parameters.h'), 'r')
        fout = open('{}/firmware/parameters.h'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():

            if '//hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(set(sum((layer.include_list for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include

            elif '//hls-fpga-machine-learning insert weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += '#include "weights/{}.h"\n'.format(w.name)

            elif "//hls-fpga-machine-learning insert layer-config" in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.config_cpp()
                    if config:
                        newline += '// ' + layer.name + '\n'
                        newline += config + '\n'
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_weights(self, model):
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(weights, model.config.get_output_dir())

    def __make_dat_file(self, original_path, project_path):
        """
        Convert other input/output data types into a dat file, which is
        a text file with the falttened matrix printed out. Note that ' ' is
        assumed to be the delimiter.
        """

        # Take in data from current supported data files
        if original_path[-3:] == "npy":
            data = np.load(original_path)
        else:
            raise Exception("Unsupported input/output data files.")

        # Faltten data, just keep first dimension
        data = data.reshape(data.shape[0], -1)

        def print_data(f):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(data[i][j]) + " ")
                f.write("\n")

        # Print out in dat file
        with open(project_path, "w") as f:
            print_data(f)

    def write_test_bench(self, model):
        ###################
        # test bench
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists('{}/tb_data/'.format(model.config.get_output_dir())):
            os.mkdir('{}/tb_data/'.format(model.config.get_output_dir()))

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == "dat":
                copyfile(input_data, '{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(input_data, '{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))

        if output_predictions:
            if output_predictions[-3:] == "dat":
                copyfile(output_predictions, '{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(output_predictions, '{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))

        f = open(os.path.join(filedir, '../templates/vivado/myproject_test.cpp'), 'r')
        fout = open('{}/{}_test.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            # Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert data' in line:
                newline = line
                offset = 0
                for inp in model.get_input_variables():
                    newline += '      ' + self.variable_definition_cpp(model, inp) + ';\n'
                    newline += '      nnet::copy_data<float, {}, {}, {}>(in, {});\n'.format(inp.type.name, offset, inp.size_cpp(), inp.cppname)
                    offset += inp.size()
                for out in model.get_output_variables():
                    newline += '      ' + self.variable_definition_cpp(model, out) + ';\n'
            elif '//hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inp in model.get_input_variables():
                    newline += '    ' + self.variable_definition_cpp(model, inp) + ';\n'
                    newline += '    nnet::fill_zero<{}, {}>({});\n'.format(inp.type.name, inp.size_cpp(), inp.cppname)
                for out in model.get_output_variables():
                    newline += '    ' + self.variable_definition_cpp(model, out) + ';\n'
            elif '//hls-fpga-machine-learning insert top-level-function' in line:
                newline = line

                size_str = indent + 'unsigned short {},{};\n'
                input_size_vars = ','.join(['size_in{}'.format(i) for i in range(1, len(model.get_input_variables()) + 1)])
                output_size_vars = ','.join(['size_out{}'.format(o) for o in range(1, len(model.get_output_variables()) + 1)])
                newline += size_str.format(input_size_vars, output_size_vars)

                input_vars = ','.join([i.cppname for i in model.get_input_variables()])
                output_vars = ','.join([o.cppname for o in model.get_output_variables()])
                top_level = indent + '{}({},{},{},{});\n'.format(model.config.get_project_name(), input_vars, output_vars, input_size_vars, output_size_vars)
                newline += top_level
            elif '//hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for out in model.get_output_variables():
                    newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(out.size_cpp())
                    newline += indent + '  std::cout << pr[i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'
            elif '//hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                for out in model.get_output_variables():
                    newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(out.type.name, out.size_cpp(), out.cppname)  # TODO enable this
            elif '//hls-fpga-machine-learning insert output' in line or '//hls-fpga-machine-learning insert quantized' in line:
                newline = line
                for out in model.get_output_variables():
                    newline += indent + 'nnet::print_result<{}, {}>({}, std::cout, true);\n'.format(out.type.name, out.size_cpp(), out.cppname)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_bridge(self, model):
        ###################
        # c++-python bridge
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/myproject_bridge.cpp'), 'r')
        fout = open('{}/{}_bridge.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        indent = '    '

        for line in f.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))
            elif '//hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join(['{type} {name}[{shape}]'.format(type=dtype, name=i.cppname, shape=i.size_cpp()) for i in model_inputs])
                outputs_str = ', '.join(['{type} {name}[{shape}]'.format(type=dtype, name=o.cppname, shape=o.size_cpp()) for o in model_outputs])
                insize_str = ', '.join(['unsigned short &const_size_in_{}'.format(i) for i in range(1, len(model_inputs) + 1)])
                outsize_str = ', '.join(['unsigned short &const_size_out_{}'.format(o) for o in range(1, len(model_outputs) + 1)])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + ',\n'
                newline += indent + insize_str + ',\n'
                newline += indent + outsize_str + '\n'
            elif '//hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for i in model_inputs:
                    newline += indent + '{var};\n'.format(var=self.variable_definition_cpp(model, i, name_suffix='_ap'))
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(dtype, i.type.name, i.size_cpp(), i.cppname, i.cppname)
                newline += '\n'

                for o in model_outputs:
                    newline += indent + '{var};\n'.format(var=self.variable_definition_cpp(model, o, name_suffix='_ap'))

                newline += '\n'

                input_size_vars = ','.join(['const_size_in_{}'.format(i) for i in range(1, len(model.get_input_variables()) + 1)])
                output_size_vars = ','.join(['const_size_out_{}'.format(o) for o in range(1, len(model.get_output_variables()) + 1)])
                input_vars = ','.join([i.cppname + '_ap' for i in model.get_input_variables()])
                output_vars = ','.join([o.cppname + '_ap' for o in model.get_output_variables()])
                top_level = indent + '{}({}, {}, {}, {});\n'.format(model.config.get_project_name(), input_vars, output_vars, input_size_vars, output_size_vars)
                newline += top_level

                newline += '\n'

                for o in model_outputs:
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(o.type.name, dtype, o.size_cpp(), o.cppname, o.cppname)
            elif '//hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    if layer.function_cpp() and model.config.trace_output and layer.get_attr('Trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += indent + 'nnet::trace_outputs->insert(std::pair<std::string, void *>("{}", (void *) malloc({} * element_size)));\n'.format(layer.name, var.size_cpp())

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_build_script(self, model):
        ###################
        # build_prj.tcl
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/vivado/build_prj.tcl'), 'r')
        fout = open('{}/build_prj.tcl'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():

            line = line.replace('myproject', model.config.get_project_name())

            if 'set_part {xcku115-flvb2104-2-i}' in line:
                line = 'set_part {{{}}}\n'.format(model.config.get_config_value('XilinxPart'))
            elif 'create_clock -period 5 -name default' in line:
                line = 'create_clock -period {} -name default\n'.format(model.config.get_config_value('ClockPeriod'))

            fout.write(line)
        f.close()
        fout.close()

        ###################
        # vivado_synth.tcl
        ###################

        f = open(os.path.join(filedir, '../templates/vivado/vivado_synth.tcl'), 'r')
        fout = open('{}/vivado_synth.tcl'.format(model.config.get_output_dir()), 'w')
        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            if '-part' in line:
                line = 'synth_design -top {} -part {}\n'.format(model.config.get_project_name(), model.config.get_config_value('XilinxPart'))

            fout.write(line)
        f.close()
        fout.close()

        ###################
        # build_lib.sh
        ###################

        f = open(os.path.join(filedir, '../templates/vivado/build_lib.sh'), 'r')
        fout = open('{}/build_lib.sh'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            line = line.replace('mystamp', model.config.get_config_value('Stamp'))

            fout.write(line)
        f.close()
        fout.close()

    def write_nnet_utils(self, model):
        ###################
        # nnet_utils
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/vivado/nnet_utils/')
        dstpath = '{}/firmware/nnet_utils/'.format(model.config.get_output_dir())

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        ###################
        # ap_types
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/vivado/ap_types/')
        dstpath = '{}/firmware/ap_types/'.format(model.config.get_output_dir())

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

    def write_tar(self, model):
        ###################
        # Tarball output
        ###################

        with tarfile.open(model.config.get_output_dir() + '.tar.gz', mode='w:gz') as archive:
            archive.add(model.config.get_output_dir(), recursive=True)

    def write_hls(self, model):
        print('Writing HLS project')
        self.write_project_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        self.write_weights(model)
        self.write_defines(model)
        self.write_parameters(model)
        self.write_test_bench(model)
        self.write_bridge(model)
        self.write_build_script(model)
        self.write_nnet_utils(model)
        self.write_tar(model)
        print('Done')
