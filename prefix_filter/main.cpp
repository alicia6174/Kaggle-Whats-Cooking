#include <stdio.h>
#include <stdlib.h>
#include <string>


using namespace std;

int main(int argc, const char * argv[])
{
    if (argc <= 1)
    {
        fprintf(stderr, "error\n");
        return -1;
    }


    FILE *fp = NULL;
    const char * file_name = argv[1];

    if ((fp = fopen(file_name, "r")) == NULL) {
        fprintf(stderr, "can't open file ");
        perror(file_name);
        exit(-1);
    }

    FILE *wfp = NULL;
    {
        const char * file_name_w =
            string(string(file_name) + ".tmp").c_str();

        if ((wfp = fopen(file_name_w, "wb+")) == NULL) {
            fprintf(stderr, "can't open file ");
            perror(file_name);
            exit(-1);
        }


    }


    int l = 1;
    while (1)
    {
        char ch = 0;


        if (fread(&ch, 1,1, fp) <= 0)
            break;

        if (ch == '\n')
            ++l;


        if (ch == (char)0xc3||
            ch == (char)0xc2||
            ch == (char)0xe2
           )
        {
            continue;
        }

        if (ch == (char)0xa7)
        {
            ch = 'c';
        }

        if (ch == (char)0xa8 ||
            ch == (char)0xa9
            )
        {
            ch = 'e';
        }

        if (ch == (char)0xae)
        {
            ch = 'i';
        }

        if (ch == (char)0xba)
        {
            ch = 'u';
        }
        unsigned char r = ch;
        if (r >= 128)
        {
            continue;
            printf("%d:%x\n", l, r);
        }

        {
            fwrite(&ch, 1, 1, wfp);

        }

    }

    fclose(fp);
    fclose(wfp);
    return 0;
}
