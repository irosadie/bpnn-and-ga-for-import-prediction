
import random
import math


class Bpnn:
    newron_n = 12
    newron_gen_n = 12
    bias_input_n = 1
    bias_input_gen_n = 12
    weight_n = 1
    weight_gen_n = 12
    bias_output_n = 1
    bias_output_gen_n = 1

    def randomData(self, n, gen_n, child_name=None):
        data = []
        i = 0
        while i < n:
            gen = []
            z = 0
            while z < gen_n:
                gen.append(round(random.uniform(0, 1), 3))
                z += 1
            data.append({f"{child_name}": gen}
                        ) if child_name else data.append(gen)
            i += 1
        return data

    def generateChromosome(self, X_train):
        i = 0
        chromosome = []
        while i < len(X_train):
            newron = self.randomData(self.newron_n, self.newron_gen_n)
            bias_input = self.randomData(
                self.bias_input_n, self.bias_input_gen_n)
            weight = self.randomData(self.weight_n, self.weight_gen_n)
            bias_output = self.randomData(
                self.bias_output_n, self.bias_output_gen_n)
            chromosome.append({"input_hidden": {"newron": newron, "bias": bias_input[0]}, "output_hidden": {
                "weight": weight[0], "bias": bias_output[0][0]}})
            i += 1
        return chromosome

    def valueOf(self, x, newron, bias):
        z = bias
        for k, v in enumerate(x):
            z += (float(v) * float(newron[k]))

        return z

    def valueOfAw(self, alfa, delta, z):
        return alfa * delta * float(z)

    def bpnn(self, chromosome, X_train, y_train, alfa):
        mse = []
        fitness = []
        # return 1
        # LOOP
        is_update_chromosom = []
        for is_key, is_val in enumerate(chromosome):
            newron = is_val['input_hidden']['newron']
            bias_input = is_val['input_hidden']['bias']
            weight = is_val['output_hidden']['weight']
            bias_output = is_val['output_hidden']['bias']
            # DATA LATIH
            y = []
            isy_train = []
            for key, val in enumerate(X_train):
                # FEED FORWARD
                isy_train.append(float(y_train[key]))
                z_in = []
                z = []

                for k, v in enumerate(newron):
                    z_in_value = self.valueOf(val, v, bias_input[k])
                    z_in.append(z_in_value)
                    z_value = 1/(1+math.exp(-z_in_value))
                    z.append(z_value)
                y_in_value = self.valueOf(z, weight, bias_output)
                y_in = y_in_value
                y_value = float(1/(1+math.exp(-y_in)))
                y.append(y_value)
                # BACK FORWARD
                init_delta = float(
                    (float(y_train[key]) - y_value)*y_value*(1-y_value))

                sigma_w_0 = (init_delta * alfa)
                sigma_w = []
                delta_in = []
                delta = []
                weight_correction = []
                bias = []
                for k, v in enumerate(z):
                    sigma_w_value = self.valueOfAw(alfa, init_delta, v)
                    sigma_w.append(sigma_w_value)
                    delta_in_value = weight[k] * init_delta
                    delta_in.append(delta_in_value)
                    delta_value = float(delta_in_value * z[k] * (1-z[k]))
                    delta.append(delta_value)
                    weight_correction_value = []
                    for i in val:
                        weight_correction_value.append(
                            alfa*delta_value*float(i))
                    weight_correction.append(weight_correction_value)

                    bias_value = float(alfa * delta_value)
                    bias.append(bias_value)
                isnewron = []
                for iskey, isval in enumerate(weight_correction):
                    isnewron_value = []
                    for xy, yz in enumerate(isval):
                        isnewron_value.append(newron[iskey][xy]+yz)
                    isnewron.append(isnewron_value)

                isbias_input = []
                for iskey, isval in enumerate(bias_input):
                    isbias_input.append(bias[iskey]+isval)

                isweight = []
                for iskey, isval in enumerate(sigma_w):
                    isweight.append(isval+weight[iskey])

                isbias_output = (sigma_w_0+bias_output)
                # Replace with new data!
                newron = isnewron
                bias_input = isbias_input
                weight = isweight
                bias_output = isbias_output
            is_update_chromosom.append({'input_hidden': {'newron': newron, 'bias': bias_input}, 'output_hidden': {
                                       'weight': weight, 'bias': bias_output}})
            # looking for fitness
            fitness_value = 0
            mse_tmp = []
            for k, i in enumerate(y):
                mse_tmp.append(pow((i - isy_train[k]), 2))
            mse_value = sum(mse_tmp)

            fitness_value = 1/mse_value
            mse.append(mse_value)
            fitness.append(fitness_value)
        return {'fitness': fitness, 'update_chromosom': is_update_chromosom}

    def testing(self, best_chromosome, X_test, y_test):
        newron = best_chromosome['input_hidden']['newron']
        bias_input = best_chromosome['input_hidden']['bias']
        weight = best_chromosome['output_hidden']['weight']
        bias_output = best_chromosome['output_hidden']['bias']
        testing = []
        for key, val in enumerate(X_test):
            z_in = []
            z = []
            for k, v in enumerate(newron):
                z_in_value = self.valueOf(val, v, bias_input[k])
                z_in.append(z_in_value)
                z_value = 1/(1+math.exp(-z_in_value))
                z.append(z_value)
            y_in_value = self.valueOf(z, weight, bias_output)
            y_in = y_in_value
            y_value = 1/(1+math.exp(-y_in))
            y = y_value
            mse = (pow(y-float(y_test[key]), 2)/2)
            testing.append(
                {'predict': y, 'target': float(y_test[key]), 'mse': mse})
        return testing
