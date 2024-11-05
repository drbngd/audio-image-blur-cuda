#ifndef PPM_H
#define PPM_H

void output_image_file(const char *filename, const uchar4 *image, int dim);
void input_image_file(const char *filename, uchar4 *image, int dim);

#endif