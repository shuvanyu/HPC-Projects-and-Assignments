# include <cstdlib>
# include <iostream>
using namespace std;

int main ();
void f (int n);

int main ()
//    TEST01 calls F, which has a memory "leak".  This memory leak can be
//    detected by VALGRID.
{
  int n = 10;
  cout << "\n";
  cout << "TEST01\n";
  cout << "  C++ version.\n";
  cout << "  A sample code for analysis by VALGRIND.\n";

  f (n);
  //
  //  Terminate.
  //
  cout << "\n";
  cout << "TEST01\n";
  cout << "  Normal end of execution.\n";

  return 0;
}

void f (int n)

//****************************************************************************80
//
//  Purpose:
//
//    F computes N+1 entries of the Fibonacci sequence.
{
  int i;
  int *x;

  x = (int*) malloc ((n+1)*sizeof(int));

  x[0] = 1;
  cout << "  " << 0 << "  " << x[0] << "\n";

  x[1] = 1;
  cout << "  " << 1 << "  " << x[1] << "\n";

  for (i=2; i<=n; i++)  /*** We have allocated a memory size of n*(sizeofint) = 10*4=40 bytes. 
                        However, when the loop i tries to run from 0 to n (which is n+1 times), we are trying to access an additional
                        4 byte of memory chunk (total 44 bytes) which we have not defined. So there can be two possible ways to solve this.
                        (1) Setting x = (int*) malloc ((n+1)*sizeof(int)); (and the rest of the code unchanged) will allocate 44 bytes 
                        of memory chunk (which we followed here). OR
                        (2) Modifying the loop value to: for (i=2; i<n; i++); which stores data from 0 to 40 bytes of memory***/
  {
    x[i] = x[i-1] + x[i-2];
    cout << "  " << i << "  " << x[i] << "\n";
  }

  free(x);          /*** With malloc one should use free and delete [] should be used with new ***/

  return;
}