#include <stdio.h>
#include "ppm.h"

void output_image_file(const char *filename, const uchar4* image, int dim) {
    FILE *f;

    //open the output file and write header info for PPM filetype
    f = fopen(filename, "wb");
    if (f == NULL){
        fprintf(stderr, "Error opening 'output.ppm' output file\n");
        exit(1);
    }
    fprintf(f, "P6\n");
    fprintf(f, "# blurred output\n");
    fprintf(f, "%d %d\n%d\n", dim, dim, 255);
    for (int x = 0; x < dim; x++){
        for (int y = 0; y < dim; y++){
            int i = x + y*dim;
            fwrite(&image[i], sizeof(unsigned char), 3, f); //only write rgb (ignoring a)
        }
    }

    fclose(f);
    printf("Image data written to %s\n", filename);
}

void input_image_file(const char *filename, uchar4* image, int dim) {
    FILE *f;

    char temp[256];
    unsigned int x, y, s;

    //open the input file and write header info for PPM filetype
    f = fopen(filename, "rb");
    if (f == NULL){
        fprintf(stderr, "Error opening 'input.ppm' input file\n");
        exit(1);
    }
    fscanf(f, "%s\n", &temp);
    fscanf(f, "%d %d\n", &x, &y);
    fscanf(f, "%d\n", &s);
    if ((x != y) && ((int)x != dim)){
        fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
        exit(1);
    }

    for (int x = 0; x < dim; x++){
        for (int y = 0; y < dim; y++){
            int i = x + y*dim;
            fread(&image[i], sizeof(unsigned char), 3, f); //only read rgb
            //image[i].w = 255;
        }
    }

    fclose(f);
    printf("Image data read from %s\n", filename);
}

