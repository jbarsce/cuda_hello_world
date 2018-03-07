# Notas Taller CUDA (Dictado por Esteban Clua, NVIDIA). 06/03/2018

## A partir de diapositivas

C�mo usar la GPU:
* Libraries (alto nivel ej. librer�a API CUDA NN o Physics).
* Compiler directives (ej. usar GPU en un loop en c�digo de alto nivel, abstray�ndonos de detalles internos de la arquitectura. Ej. de librer�a: Open ACC).
* Programming language (CUDA, OpenCL).

Notaci�n:

* A la GPU le llamamos Device.
* A todo lo externo a la GPU, es decir memoria RAM, procesador, disco r�gido

Algunas ideas sueltas:

* Debemos tener en cuenta que se gasta m�s energ�a con el proceso de copia de Device a Host que en el procesamiento dentro del Host.
* Para paralelizar, no podemos depender de que un thread se ejecute antes que otro.
* En la programaci�n de GPU, m�s que disminuir latencia lo que se tiende es a maximizar la tasa de ocupaci�n de la GPU.
* La CPU es siempre quien debe disparar el kernel. Al hace eso establece el n�mero de bloques y cu�l es el tama�o de cada bloque.
* La llamada de un kernel por parte de la CPU es asincr�nica, es decir que la CPU puede seguir con sus instrucciones. Entonces, cuando la CPU necesite de los datos procesados por la GPU, se necesita utilizar una sincronizaci�n.
* Una estrategia muy com�n para paralelizar es cambiando "for" por threadIdx. La idea es utilizar N threads, donde cada thread es una iteraci�n.
* Qu� sucede cuando $N > 1024$ (es decir que necesitemos m�s threads de bloques)? Se aumenta la carga de trabajo de cada thread, haciendo un loop en cada thread.
* �Cu�ndo vale la pena utilizar una memoria compartillada? "Cuando el dato lo tenga que usar m�s de una vez, ejemplo en multiplicaci�n de matrices. Si s�lo proceso al leer y escribir en memoria global no vale la pena". Para utilizar la memoria compartirllada se usa: 

        __shared__ float temp_data[256]

Disparidad -> cuando grupo de n�cleos hace una tarea y los dem�s quedan ociosos.


GRID -> threads que est�n siendo ejecutados con respecto a alg�n kernel. Es generado cuando se ejecuta un kernel.


Kernels concurrentes -> kernels distintos corriendo en paralelo con distintas tareas (esto es dif�cil de optimizar)


Thread -> ejecuci�n de un kernel con un procesador. Posee un id propio. El n�mero de threads no necesariamente coincide con el n�mero de cores.


Memoria global (Main memory) -> memoria grande principal de la GPU (aprox. mismo orden de grandeza que una CPU). Memoria fuera del procesador, "lenta", tiene comunicaci�n con el host read/write.


Memoria constante -> memoria read-only.


CUDA (c�digo de programa), PTX (c�digo CUDA compilado) y Cubin (runtime de c�digo PTX adaptado al tipo de Hardware)


Streaming Multiprocessors (SM) -> Grupos de cores que tienen en com�n la "unidad de control". Comparten una memoria peque�a muy r�pida (memoria compartillada) entre grupos de cores, pudiendo todos ellos leer y escribir datos.


Bloques -> grupos de threads que son ejecutados en un mismo SM. El n�mero m�ximo de SMs que se puede ejecutar dentro de un bloque es de 1024, independientemente de la arquitectura. La cantidad de threads en un grid se obtiene multiplicando el n�mero de bloques con la cantidad de bloques. Los threads diferentes no tienen forma de comunicarse entre ellos m�s que por la memoria global de la GPU. 32 es el "n�mero m�gico " de los bloques; optimizar en 32 o m�ltiplo de 32. Si se crea un bloque con 32 threads, todos los mismos se van a ocupar al mismo tiempo. Dentro de cada bloque comparten una shared memory, que es una memoria r�pida pero temporal. Cada bloque a su vez posee registradores para guardado de variables. El bloque puede ser "dibujado" (interpretado) como de 1, 2 o 3 dimensiones (ver memoria coalescente para tener cuidado). Existe un l�mite de threads por bloque, dependiendo de las compute capability de cada versi�n de CUDA. Por otra parte, no existe un l�mite en la cantidad de bloques ya que son unidades virtuales de trabajo.


Warps -> secuencia de 32 datos que vamos a ir procesando en cada bloque (a�n considerando que cada bloque puede tener 1024 threads). Cada secuencia de 32 threads tiene un controlador de Warp propio, realizado por la unidad de Warp scheduler. La disparidad es un problema cuando existe disparidad en los warps.


Memoria coalescente -> Cuando se busca un dato en la memoria, la GPU lee ese dato y los 124 bytes siguientes. Es decir que si el thread 0 pide el dato que est� en 0, el 1 el del 1, etc., se obtendr�n todos los datos pedidos en una �nica b�squeda (400 ciclos de m�quina). Esto se optimiza haciendo que los threads del mismo warp b�squen datos parecidos. Este problema no es �nico de las GPUs sino del parelelismo.


Operaciones at�micas. �Qu� pasa si dos threads deben leer y escribir en el mismo espacio de la memoria global? Se puede utilizan iteraciones at�micas, en donde un thread accede y le limita el acceso a los dem�s. Cuidado porque puede generar overhead y, eventualmente, hacer una cola de acceso a la memoria y terminar con el paralelismo.


Streams overlaps: siendo la memoria global cara de acceder, el c�digo puede optimizarse utilizando "streams" de escritura (donde se escribe lo m�nimo e indispensable para poder ejecutarse), se ejecuta, se escribe, ....

## Pseudoc�digo

		//la directiva _global_ indica que no debe ser compilado para la CPU sino para la GPU, compilando un PTX
		_global_ void mykernel(void) { 
		}

		int main(void) {
			//se establece el n�mero de bloques necesario y el tama�o de cada bloques en <<1,1>>
			//la cantidad de "dimensiones" est� dada por el n�mero de bloques necesarios
			//S�lo puede haber 3 dimensiones (x, y y z)
			//Por otra parte, 1 de tama�o es un n�mero sub-�ptimo para el tama�o de bloque porque no es m�ltiplo de 32
			mykernel<<1,1>>>(); 
			printf("Hello world \n");
			return 0;
		}

Otro ejemplo

		# en este c�digo se le pasa informaci�n a la GPU
		_global_void add(int *a, int *b, int *c){
		*c = *a + *b;
		}


Otro... (suma de vectores)

		__global__ void add(int) {
			int i = threadIdx.x + blockIdx.x * blockDim.x;
			if (i < N) {
				d_c[i] = d_a[i] + d_b[i];
			}


			int main() {
				vecAdd <<K,M>>>(A, B, C);	// K*M >= N
			}
		}


Ejercicios sugeridos:

1. Extender el vector en suma de n�meros y luego convertirlo en matriz.
2. Reducci�n de un vector: sacar el promedio de los elementos de un vector (utilizando un algortmo de paralelizaci�n).

## M�todos y variables �tiles

		Malloc() - en CUDA es- cudaMalloc()   // hace reserva de memoria
		Free() - en CUDA es- cudaFree()  // se libera el espacio de memoria sin uso. Al no haber un SO en la GPU, es muy importante
		memcpy() -en CUDA es- cudaMemcpy()  // permite intercambiar datos entre la CPU y GPU
		cudaMallocManaged()  // incorpora la memoria global
		cudaDeviceSynchronize()  // punto de sincronizaci�n, la CPU se queda esperando a que finalice el procesamiento de la GPU
		threadIdx // es el n�mero de id del thread, establecido por la GPU. Normalmente el paralelismo del problema se mapea con el threadIdx
		__syncthread()  // sincroniza todos los threads de un mismo bloque. Importante: no debe haber disparidad, o podr�a ocurrir un deadlock

Para sumarle al id del bloque y procesar (ejemplo el sub-�ndice de cada vector) se usa

		int i= threadIdx.x + blockIdx.x * blockDim.x;


## Compilar un programa en CUDA

		1. Name file as .cu
		2. Nvcc name.cu
		3. ./a.out


## Debuggers

* NSight. Muy recomendada
* CUDA GDB
* CUDA Memcheck
* Nvprof ./a.out (testea y mira estad�sticas de cada tabla)
