/*

    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <omp.h>
#include <math.h>
#include "matrix_mul.h"

#define CORE_COUNT 8
#define SEQUNTIAL 4

namespace omp
{
    void
    matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
    {
        float sub_array_1[sq_dimension], sub_array_2[sq_dimension], sub_array_3[sq_dimension], sub_array_4[sq_dimension], local_product[SEQUNTIAL];
        #pragma omp parallel num_threads(CORE_COUNT) private(sub_array_1, sub_array_2, sub_array_3, sub_array_4, local_product)
        {
            int id = omp_get_thread_num();
            int nthrds = omp_get_num_threads();  
            for (unsigned int i = (SEQUNTIAL*id); i < sq_dimension; i+=nthrds)
            {
                if (sq_dimension - i > 3)
                {
                    for (unsigned int j = 0; j < sq_dimension; j++)
                    {
                        sub_array_1[j] = sq_matrix_2[j*sq_dimension + i + 0];
                        sub_array_2[j] = sq_matrix_2[j*sq_dimension + i + 1];
                        sub_array_3[j] = sq_matrix_2[j*sq_dimension + i + 2];
                        sub_array_4[j] = sq_matrix_2[j*sq_dimension + i + 3];
                    }

                    for (unsigned int k = 0; k < sq_dimension; k++)
                    {
                        local_product[0] = 0.0;
                        local_product[1] = 0.0;
                        local_product[2] = 0.0;
                        local_product[3] = 0.0;

                        for (unsigned int l = 0; l < sq_dimension; l++)
                        {
                            local_product[0] += sub_array_1[l] * sq_matrix_1[k*sq_dimension + l];
                            local_product[1] += sub_array_2[l] * sq_matrix_1[k*sq_dimension + l];
                            local_product[2] += sub_array_3[l] * sq_matrix_1[k*sq_dimension + l];
                            local_product[3] += sub_array_4[l] * sq_matrix_1[k*sq_dimension + l];
                        }

                        sq_matrix_result[k*sq_dimension + i + 0] = local_product[0];
                        sq_matrix_result[k*sq_dimension + i + 1] = local_product[1];
                        sq_matrix_result[k*sq_dimension + i + 2] = local_product[2];
                        sq_matrix_result[k*sq_dimension + i + 3] = local_product[3];
                    }
                } 
                else if (sq_dimension - i == 3) 
                {
                    for (unsigned int j = 0; j < sq_dimension; j++)
                    {
                        sub_array_1[j] = sq_matrix_2[j*sq_dimension + i + 0];
                        sub_array_2[j] = sq_matrix_2[j*sq_dimension + i + 1];
                        sub_array_3[j] = sq_matrix_2[j*sq_dimension + i + 2];
                    }

                    for (unsigned int k = 0; k < sq_dimension; k++)
                    {
                        local_product[0] = 0.0;
                        local_product[1] = 0.0;
                        local_product[2] = 0.0;

                        for (unsigned int l = 0; l < sq_dimension; l++)
                        {
                            local_product[0] += sub_array_1[l] * sq_matrix_1[k*sq_dimension + l];
                            local_product[1] += sub_array_2[l] * sq_matrix_1[k*sq_dimension + l];
                            local_product[2] += sub_array_3[l] * sq_matrix_1[k*sq_dimension + l];
                        }

                        sq_matrix_result[k*sq_dimension + i + 0] = local_product[0];
                        sq_matrix_result[k*sq_dimension + i + 1] = local_product[1];
                        sq_matrix_result[k*sq_dimension + i + 2] = local_product[2];
                    }
                }
                else if (sq_dimension - i == 2) 
                {
                    for (unsigned int j = 0; j < sq_dimension; j++)
                    {
                        sub_array_1[j] = sq_matrix_2[j*sq_dimension + i + 0];
                        sub_array_2[j] = sq_matrix_2[j*sq_dimension + i + 1];
                    }

                    for (unsigned int k = 0; k < sq_dimension; k++)
                    {
                        local_product[0] = 0.0;
                        local_product[1] = 0.0;

                        for (unsigned int l = 0; l < sq_dimension; l++)
                        {
                            local_product[0] += sub_array_1[l] * sq_matrix_1[k*sq_dimension + l];
                            local_product[1] += sub_array_2[l] * sq_matrix_1[k*sq_dimension + l];
                        }

                        sq_matrix_result[k*sq_dimension + i + 0] = local_product[0];
                        sq_matrix_result[k*sq_dimension + i + 1] = local_product[1];
                    }
                }
                else if (sq_dimension - i == 1) 
                {
                    for (unsigned int j = 0; j < sq_dimension; j++)
                    {
                        sub_array_1[j] = sq_matrix_2[j*sq_dimension + i + 0];
                    }

                    for (unsigned int k = 0; k < sq_dimension; k++)
                    {
                        local_product[0] = 0.0;

                        for (unsigned int l = 0; l < sq_dimension; l++)
                        {
                            local_product[0] += sub_array_1[l] * sq_matrix_1[k*sq_dimension + l];
                        }

                        sq_matrix_result[k*sq_dimension + i + 0] = local_product[0];
                    }
                }
            }
        }
    }
} //namespace omp