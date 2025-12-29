/*
** svn $Id: sed_toy.h 2232 2012-01-03 18:55:20Z arango $
*******************************************************************************
** Copyright (c) 2002-2012 The ROMS/TOMS Group                               **
**   Licensed under a MIT/X style license                                    **
**   See License_ROMS.txt                                                    **
*******************************************************************************
**
** Options for One-Dimensional (vertical) Sediment Toy.
**
** Application flag:   BEAUFORT_RIVERS_AK_003
** Input scripts:      ocean_beaufort_ak_003.in
**                     sediment_beaufort_ak_003_13rivs.in
*/

#define ROMS_MODEL

#undef  BODYFORCE
#undef  LOG_PROFILE
#define DJ_GRADPS
#define SALINITY
#undef SPLINES_VVISC
#undef SPLINES_VDIFF
#define OUT_DOUBLE
#define AVERAGES
#define PERFECT_RESTART

/* #define MPDATA or MP_DATA */
/* Tracer Advection */
#define TS_MPDATA
#define TS_MPDATA_LIMIT
#define NONLIN_EOS

#undef HADVECTION
#undef VADVECTION

/* Boundary Conditions */ 
#define RADIATION_2D
#undef ANA_FSOBC
#undef ANA_M2OBC
#undef ANA_M3OBC
#undef ANA_TOBC

#undef ANA_GRID
#undef ANA_INITIAL

#define SOLVE3D
#ifdef SOLVE3D
# undef ANA_SEDIMENT
# define ANA_BPFLUX
# define ANA_BSFLUX
# define ANA_BTFLUX
# define ANA_SPFLUX
# define ANA_SRFLUX
# undef ANA_SSFLUX
# undef ANA_STFLUX
#endif
#undef  ANA_VMIX

#undef  ANA_WWAVE
#ifdef ANA_WWAVE
# define BBL_MODEL
# define WAVES_HEIGHT
# define WAVES_LENGTH
# define WAVES_BOT_PERIOD
#endif

#define MASKING
#undef WET_DRY

#undef BULK_FLUXES
#ifdef BULK_FLUXES
# undef ANA_TAIR
# undef ANA_PAIR
# undef ANA_HUMIDITY
# undef ANA_WINDS
# define LONGWAVE
# define ANA_CLOUD
# define ANA_RAIN
#else
# undef ANA_SMFLUX
# undef ANA_STFLUX
#endif

/* Advection schemes */
#define UV_ADV
#define TS_C4VADVECTION
#define TS_A4HADVECTION 

/* Viscosity */
#define UV_VIS2
#define VISC_GRID
#define UV_COR
#define MIX_GEO_UV

/* Diffusivity */
#define TS_DIF2
#define DIFF_GRID
#define MIX_GEO_TS

/* select one of six bottom stress methods */
#undef UV_LOGDRAG
#undef  UV_LDRAG
#undef  UV_QDRAG
#undef  SG_BBL
#undef  MB_BBL
#define SSW_BBL

#ifdef SG_BBL
# undef  SG_CALC_ZNOT
# undef  SG_LOGINT
#endif
#ifdef MB_BBL
# undef  MB_CALC_ZNOT
# undef  MB_Z0BIO
# undef  MB_Z0BL
# undef  MB_Z0RIP
#endif
#ifdef SSW_BBL
# define SSW_CALC_ZNOT
# undef  SSW_LOGINT
#endif

/* turb closure */
#undef GLS_MIXING
#ifdef GLS_MIXING
# define KANTHA_CLAYSON
# define N2S2_HORAVG
# define RI_SPLINES
# undef  CRAIG_BANNER
# undef  CHARNOK
# undef  ZOS_HSIG
# undef  TKE_WAVEDISS
#endif
#define MY25_MIXING
#define N2S2_HORAVG

/* sediment choices */
#define SEDIMENT
#ifdef SEDIMENT
# define SUSPLOAD
# undef  BEDLOAD_SOULSBY
# undef  BEDLOAD_MPM
# define SED_DENS
# define SED_MORPH
# undef  SED_BIODIFF
# undef  NONCOHESIVE_BED1
# undef  NONCOHESIVE_BED2
# define COHESIVE_BED
# undef  MIXED_BED
#endif


