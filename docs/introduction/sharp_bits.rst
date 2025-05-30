‼️ The Sharp Bits of NNTool
###########################

Although the ``nntool`` library is designed to be simple and easy to use, there are some sharp bits that you should be aware of. This section covers some of the sharp bits that you should be aware of when using the ``nntool`` library.

‼️ Pure functions
==============

The submitted function should be a pure function. This means that the function should not have any side effects.

.. tip::

   To be a pure function, the function should satisfy the following conditions:

   - The function should not modify any global variables or modify any external state.
   - The function should only depend on the input arguments and should return the output based on the input arguments.


‼️ Stateless submissions
==========================

Let's say we submitted a job with code version 1.0 and the job is pending due to resource constraints. After that, we made some changes to the code and get the code version 1.1. Now, if the job is executed, it would be executed with the latest code version 1.1 not with the code version 1.0.

.. warning::
   When a job is submitted, it doesn't mean it would save the current code state. The code is executed based on the latest code in the repository.

.. tip::
   To work around this issue, you can save the code in submission and execute the code based on the saved code. This way, you can ensure that the code is executed based on the code version you submitted. This can be achieved by setting ``pack_code=True`` and ``use_packed_code=True`` in the ``SlurmConfig`` function.
