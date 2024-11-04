	.intel_syntax noprefix

	.text
        //
        // function prototype:
        // unsigned long long test_amx_int8(void* tilecfg, int loops)
	.globl	amx_int8
amx_int8:
        ldtilecfg [rdi]
        test    rsi, rsi
        tilezero tmm0
        tilezero tmm1
        tilezero tmm2
        jz      _amx_int8_exit
_amx_int8_loop:
        tdpbssd tmm0, tmm1, tmm2
        //tdpbf16ps tmm0, tmm1, tmm2
        sub     rsi, 0x1
        jnz     _amx_int8_loop
_amx_int8_exit:
	mov	rax, QWORD PTR[rdi]
        tilerelease
	ret
	.size	amx_int8, .-amx_int8

	.global get_amx_xcr_reg
	// int get_amx_xcr_reg();
get_amx_xcr_reg:
	push rcx
	push rdx
	xor  rcx, rcx
	xgetbv
	pop  rdx
	pop  rcx
	ret

	.globl	amx_int8_mul
        //void * amx_int8_mul(void* cfg, void *ma, void* mb, int64 a_stride, int64 b_stride, void* mc);
        // if cfg is 0, no TILE config will be load
        // args regs: rdi, rsi, rdx, rcx, r8, r9
amx_int8_mul:
        test rdi, rdi
        jz _amx_int8_mul_skip_cfg
        ldtilecfg [rdi]
_amx_int8_mul_skip_cfg:
        tileloadd tmm0, [rsi+rcx*1]
        tileloadd tmm1, [rdx+r8*1]
        tdpbuud tmm2, tmm0, tmm1
        tilestored [r9+r8*1], tmm2
        mov       rax, r9
        tilerelease
	ret
	.size	amx_int8_mul, .-amx_int8_mul


