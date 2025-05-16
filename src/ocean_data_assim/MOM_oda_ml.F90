!> Interfaces for MOM6 ensembles and data assimilation.
module MOM_oda_ml_mod

! This file is part of MOM6. see LICENSE.md for the license.

! MOM infrastructure
use MOM_cpu_clock, only : cpu_clock_begin, cpu_clock_end, cpu_clock_id
use MOM_verticalGrid, only : verticalGrid_type
use gsw_mod_toolbox, only : gsw_ct_from_pt, gsw_sigma0
use netcdf, only : nf90_open, nf90_inq_varid, nf90_get_var, nf90_close, nf90_close
use netcdf, only : nf90_nowrite, nf90_noerr

implicit none ; private

public :: oda_ml_init, oda_ml_end, oda_ml_inference

! Data structure to save the ML configuration, input, and output data
type, public :: ocean_oda_ml_config ; private
    character(len=255)  :: filename
    real, dimension(32,34)  :: l1_weight
    real, dimension(32,32)  :: l2_weight
    real, dimension(8,32)  :: l3_weight
    real, dimension(32) :: l1_bias, l2_bias, attn_bias1
    real, dimension(8) :: l3_bias
    real, dimension(16) :: e1_bias1, e2_bias1, e3_bias1
    real, dimension(8) :: e1_bias2, e2_bias2, e3_bias2, d_bias2
    real, dimension(34) :: attn_bias2
    real, dimension(128) :: d_bias1
    real :: d_bias3
    real, dimension(16,1,3)  :: e1_weight1, e2_weight1, e3_weight1
    real, dimension(8,16,3)  :: e1_weight2, e2_weight2, e3_weight2
    real, dimension(32,34)  :: attn_weight1
    real, dimension(34,32)  :: attn_weight2
    real, dimension(128,8)  :: d_weight1
    real, dimension(16,8,3)  :: d_weight2
    real, dimension(1,8,3)  :: d_weight3

    real, dimension(:), allocatable :: z_l
    real, dimension(:), allocatable :: z_i
    integer :: nk
end type ocean_oda_ml_config

type, public :: ocean_oda_ml_data
    integer :: nk
    real :: dyCu_left, dyCu_right, dxCv_south, dxCv_north, areacello
    real :: dxCu_left, dxCu_right, dyCv_north, dyCv_south
    real :: areacellobu_left_south, areacellobu_left_north, areacellobu_right_south, areacellobu_right_north
    real :: dxCu_left_south, dxCu_left_north, dxCu_right_south, dxCu_right_north
    real :: dyCv_north_left, dyCv_north_right, dyCv_south_left, dyCv_south_right
    real :: bathyT
    real, dimension(3,3) :: all_bathy
    real :: mask2dT, OBCmaskCu_left, OBCmaskCu_right, OBCmaskCv_south, OBCmaskCv_north
    real :: mask2dT_left, mask2dT_right, mask2dT_north, mask2dT_south
    real :: OBCmaskCu_left_south, OBCmaskCu_left_north, OBCmaskCu_right_south, OBCmaskCu_right_north
    real :: OBCmaskCv_south_left, OBCmaskCv_south_right, OBCmaskCv_north_left, OBCmaskCv_north_right
    real :: geoLatT

    !! Input features
    real :: SSH !<sea surface height (m) across ensembles
    real :: taux_left !<zonal wind stress
    real :: taux_right !<zonal wind stress
    real :: tauy_north !<meridional wind stress
    real :: tauy_south !<zonal wind stress
    real :: latent !<latent heat flux
    real :: sensible !<sensile heat flux
    real :: lw !<longwave radiation flux
    real :: sw !<shortwave radiation flux
    real, pointer, dimension(:) :: T=>NULL() !<layer potential temperature (degC) across ensembles
    real, pointer, dimension(:) :: S=>NULL() !<layer salinity (psu or g kg-1) across ensembles
    real, pointer, dimension(:) :: U_left=>NULL() !<layer zonal velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: U_right=>NULL() !<layer zonal velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: V_north=>NULL() !<layer meridional velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: V_south=>NULL() !<layer meridional velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: T_left=>NULL() !<layer potential temperature (degC) across ensembles
    real, pointer, dimension(:) :: T_right=>NULL() !<layer potential temperature (degC) across ensembles
    real, pointer, dimension(:) :: T_south=>NULL() !<layer potential temperature (degC) across ensembles
    real, pointer, dimension(:) :: T_north=>NULL() !<layer potential temperature (degC) across ensembles
    real, pointer, dimension(:) :: S_left=>NULL() !<layer salinity (psu or g kg-1) across ensembles
    real, pointer, dimension(:) :: S_right=>NULL() !<layer salinity (psu or g kg-1) across ensembles
    real, pointer, dimension(:) :: S_south=>NULL() !<layer salinity (psu or g kg-1) across ensembles
    real, pointer, dimension(:) :: S_north=>NULL() !<layer salinity (psu or g kg-1) across ensembles
    real, pointer, dimension(:) :: U_left_south=>NULL() !<layer zonal velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: U_left_north=>NULL() !<layer zonal velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: U_right_south=>NULL() !<layer zonal velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: U_right_north=>NULL() !<layer zonal velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: V_north_left=>NULL() !<layer meridional velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: V_north_right=>NULL() !<layer meridional velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: V_south_left=>NULL() !<layer meridional velocity (m s-1) across ensembles
    real, pointer, dimension(:) :: V_south_right=>NULL() !<layer meridional velocity (m s-1) across ensembles
    real :: taux_left_south, taux_left_north !<zonal wind stress
    real :: taux_right_south, taux_right_north !<zonal wind stress
    real :: tauy_north_left, tauy_north_right !<meridional wind stress
    real :: tauy_south_left, tauy_south_right !<zonal wind stress

    !! Output predictions
    real, pointer, dimension(:) :: T_inc=>NULL()
    real, pointer, dimension(:) :: S_inc=>NULL()

end type ocean_oda_ml_data

real :: PRHO_change = 0.03
real :: reference_depth = 10
!real :: value_for_control_depth = 1E6
!real :: value_for_control_PRHO = 999
!real :: value_for_control_oceanzvars = 1E5
real :: ReLU_zero = 0
real, dimension(15) :: target_sigmas = (/0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9/)
real, dimension(16) :: output_flux_sigmas = (/0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0/)
character(len=255)  :: danni_ANN_name = '/gpfs/f5/gfdl_sd/world-shared/Danni.Du/ECDA_data/ML/M8/M8_ens_attention_encoder_decoder_2003_2014_30epoch_L1.nc'
real :: seconds_in_30_days = 3600*24*30

integer :: id_clock_ml_remapping
integer :: id_clock_ml_normalization
integer :: id_clock_ml_inference

#include <MOM_memory.h>

character(len=40)  :: mdl = "MOM_oda_ml" !< This module's name.

contains

    subroutine oda_ml_inference(ml_config,ml_data)
        type(ocean_oda_ml_config), pointer, intent(in) :: ml_config
        type(ocean_oda_ml_data), pointer, intent(in) :: ml_data
        
        real :: SA, PT, CT, PRHO ,tauamp, rho0
        real, dimension(:), allocatable :: PRHO_profile
        real :: PRHO_mld, PRHO_10m 
        real :: mld_depth
        integer :: zl_index_mld, zl_index10m, zl_index_3mld, right_index
        real, dimension(:), allocatable :: zl_to_sigma, zi_to_sigma
        real :: thetao, so, uo_left, uo_right, vo_south, vo_north, div, thetao_top, thetao_bottom, so_top,so_bottom
        real :: PRHO_top, PRHO_bottom, uo_right_top, uo_right_bottom, uo_left_top, uo_left_bottom
        real :: vo_north_top, vo_north_bottom, vo_south_top, vo_south_bottom
        real, dimension(15) :: thetao_zgrad_sigma, so_zgrad_sigma, PRHO_zgrad_sigma, div_sigma, output_DT_sigmas, uo_zgrad_sigma, vo_zgrad_sigma, shear2_sigma
        real :: thetao_zgrad_sigma_dist, PRHO_zgrad_sigma_dist, div_sigma_dist, shear2_sigma_dist, coef
        real, dimension(:), allocatable :: thetao_zgrad_profile, so_zgrad_profile, div_profile, PRHO_zgrad_profile, uo_zgrad_profile, vo_zgrad_profile
        real, dimension(55) :: ANN_input
        real, dimension(34) :: ANN_input_final, exp_x, attns
        real, dimension(8) :: encoder_output
        real, dimension(:), allocatable :: output_DT_at_zl, output_flux_at_zi
        real, dimension(:), allocatable :: z_l
        real, dimension(32) :: l1_output, l2_output, attns1
        real, dimension(8) :: l3_output
        real, dimension(16) :: flux_output
        real, dimension(128) :: d_output1
        real, dimension(16,8) :: d_output1_reshaped
        real, dimension(8,16) :: decoder_output
        integer :: zz, i, j, idx, k, l
        real :: mask_Tuv
        real :: Smin, sum_exp
        real :: pi
        real, dimension(17) :: all_mask


        
        
        ml_data%T_inc=0.0

        rho0 = 1035
        pi = acos(-1.0)
        
        mask_Tuv = ml_data%mask2dT + ml_data%OBCmaskCu_left + ml_data%OBCmaskCu_right + ml_data%OBCmaskCv_south + ml_data%OBCmaskCv_north
        Smin = MINVAL(ml_data%S)
        if (mask_Tuv < 5.0) then
            ml_data%T_inc=0.0
        elseif (Smin < 0.0) then
            ml_data%T_inc=0.0
        elseif (ml_data%T(1) < -0.054*ml_data%S(1)) then
            ml_data%T_inc=0.0
        else
            allocate(z_l(ml_config%nk),source=0.0)
            z_l = ml_config%z_l

            allocate(PRHO_profile(ml_data%nk),source=0.0)
            do zz  = 1, ml_data%nk
                SA = ml_data%S(zz)
                PT = ml_data%T(zz)
                CT = gsw_ct_from_pt(SA, PT)
                PRHO = gsw_sigma0(SA, CT)
                PRHO_profile(zz) = PRHO
            end do

            ! find 3 MLD
            ! first find the first index below 10m
            call find_right_index(z_l, reference_depth, ml_data%bathyT, z_l, zl_index10m)
            ! if zl_index10m is not found or it is 1, set mld_depth to be bathyT, so that the ml inference will not be done
            if (zl_index10m <= 1) then
                mld_depth = ml_data%bathyT
            else
                ! the 10m potential density
                call interpolate(z_l(zl_index10m-1),z_l(zl_index10m),PRHO_profile(zl_index10m-1), PRHO_profile(zl_index10m),reference_depth,PRHO_10m)
                ! the MLD potential density
                PRHO_mld = PRHO_10m + PRHO_change
                ! the first z_l index below MLD
                call find_right_index(PRHO_profile, PRHO_mld, ml_data%bathyT, z_l, zl_index_mld)
                ! if zl_index10m is not found or it is 1, set mld_depth to be bathyT, so that the ml inference will not be done
                if (zl_index_mld <= 1) then
                    mld_depth = ml_data%bathyT
                else
                    ! the MLD depth
                    call interpolate(PRHO_profile(zl_index_mld-1),PRHO_profile(zl_index_mld),z_l(zl_index_mld-1),z_l(zl_index_mld),PRHO_mld,mld_depth)
                    ! the MLD must be below 10m
                    if (mld_depth < 10) then
                        mld_depth = 10
                    end if
                end if
            end if

            ! the first z_l index below 3MLD
            call find_right_index(z_l(1:ml_config%nk-1), 3*mld_depth, ml_data%bathyT, z_l, zl_index_3mld)

            if (zl_index_3mld == 0) then ! if 3 mld not found
                ml_data%T_inc=0.0
            else
                if (z_l(zl_index_3mld+1) > MINVAL(ml_data%all_bathy) then
                    ml_data%T_inc=0.0
                else ! if above all bathy, then get the vertical profiles

                    zi_to_sigma = ml_config%z_i(2:zl_index_3mld + 1)/mld_depth
                    allocate(thetao_zgrad_profile(zl_index_3mld),source=0.0)
                    !allocate(so_zgrad_profile(zl_index_3mld),source=0.0)
                    allocate(PRHO_zgrad_profile(zl_index_3mld),source=0.0)
                    allocate(uo_zgrad_profile(zl_index_3mld),source=0.0)
                    allocate(vo_zgrad_profile(zl_index_3mld),source=0.0)

                    do zz = 1, zl_index_3mld
                        thetao_top = ml_data%T(zz)
                        so_top = ml_data%S(zz)
                        CT = gsw_ct_from_pt(so_top,thetao_top)
                        PRHO_top = gsw_sigma0(so_top,CT)
                        uo_right_top = ml_data%U_right(zz)
                        uo_left_top = ml_data%U_left(zz)
                        vo_north_top = ml_data%V_north(zz)
                        vo_south_top = ml_data%V_south(zz)
                    
                        thetao_bottom = ml_data%T(zz+1)
                        so_bottom = ml_data%S(zz+1)
                        CT = gsw_ct_from_pt(so_bottom,thetao_bottom)
                        PRHO_bottom = gsw_sigma0(so_bottom,CT)
                        uo_right_bottom = ml_data%U_right(zz+1)
                        uo_left_bottom = ml_data%U_left(zz+1)
                        vo_north_bottom = ml_data%V_north(zz+1)
                        vo_south_bottom = ml_data%V_south(zz+1)

                        thetao_zgrad_profile(zz) = (thetao_top - thetao_bottom)/(z_l(zz+1) - z_l(zz))
                        !so_zgrad_profile(zz) = (so_top - so_bottom)/(z_l(zz+1) - z_l(zz))
                        PRHO_zgrad_profile(zz) = (PRHO_top - PRHO_bottom)/(z_l(zz+1) - z_l(zz))
                        uo_zgrad_profile(zz) = (uo_right_top-uo_right_bottom+uo_left_top-uo_left_bottom)/(2*(z_l(zz+1) - z_l(zz)))
                        vo_zgrad_profile(zz) = (vo_north_top-vo_north_bottom+vo_south_top-vo_south_bottom)/(2*(z_l(zz+1) - z_l(zz)))
                    end do

                    zl_to_sigma = z_l(1:zl_index_3mld)/mld_depth

                    allocate(div_profile(zl_index_3mld),source=0.0)

                    !do zz = 1, zl_index_3mld
                        !call compute_current_divergence(ml_data%U_left(zz)*ml_data%dyCu_left, ml_data%U_right(zz)*ml_data%dyCu_right, &
                                !ml_data%V_south(zz)*ml_data%dxCv_south, ml_data%V_north(zz)*ml_data%dxCv_north, &
                                !ml_data%areacello, div)
                        !div_profile(zz) = div
                    !end do 
            
            
                    ! interpolate values to target_sigmas
                    do i = 1, 15
                        call find_right_index_clean(zi_to_sigma, target_sigmas(i), right_index)
                        ! quality control done in the previous steps, so that right_index >=1 and right_index <= zl_index_3mld
                        if (right_index == 1) then
                            thetao_zgrad_sigma(i) = thetao_zgrad_profile(1)
                            !so_zgrad_sigma(i) = so_zgrad_profile(1)
                            PRHO_zgrad_sigma(i) = PRHO_zgrad_profile(1)
                            uo_zgrad_sigma(i) = uo_zgrad_profile(1)
                            vo_zgrad_sigma(i) = vo_zgrad_profile(1)
                        else
                            call interpolate(zi_to_sigma(right_index-1),zi_to_sigma(right_index),thetao_zgrad_profile(right_index-1), &
                                    thetao_zgrad_profile(right_index),target_sigmas(i),thetao_zgrad_sigma(i))
                            !call interpolate(zi_to_sigma(right_index-1),zi_to_sigma(right_index),so_zgrad_profile(right_index-1), &
                                    !so_zgrad_profile(right_index),target_sigmas(i),so_zgrad_sigma(i))
                            call interpolate(zi_to_sigma(right_index-1),zi_to_sigma(right_index),PRHO_zgrad_profile(right_index-1), &
                                    PRHO_zgrad_profile(right_index),target_sigmas(i),PRHO_zgrad_sigma(i))
                            call interpolate(zi_to_sigma(right_index-1),zi_to_sigma(right_index),uo_zgrad_profile(right_index-1), &
                                    uo_zgrad_profile(right_index),target_sigmas(i),uo_zgrad_sigma(i))
                            call interpolate(zi_to_sigma(right_index-1),zi_to_sigma(right_index),vo_zgrad_profile(right_index-1), &
                                    vo_zgrad_profile(right_index),target_sigmas(i),vo_zgrad_sigma(i))
                        end if

                        !call find_right_index_clean(zl_to_sigma, target_sigmas(i), right_index)
                        !if (right_index == 1) then
                            !div_sigma(i) = div_profile(1)
                        !else
                            !call interpolate(zl_to_sigma(right_index-1),zl_to_sigma(right_index),div_profile(right_index-1),div_profile(right_index),target_sigmas(i),div_sigma(i))
                        !end if
                    end do
                    
                    tauamp = sqrt(((ml_data%taux_left+ml_data%taux_right)/2)**2+((ml_data%tauy_south+ml_data%tauy_north)/2)**2)

                    ! subroutine(input,DA tendency)
                    thetao_zgrad_sigma_dist = sqrt(sum(thetao_zgrad_sigma**2))
                    PRHO_zgrad_sigma_dist = sqrt(sum(PRHO_zgrad_sigma**2))
                    !div_sigma_dist = sqrt(sum(div_sigma**2))
                    shear2_sigma = uo_zgrad_sigma**2 + vo_zgrad_sigma**2
                    shear2_sigma_dist = sqrt(sum(shear2_sigma**2))
                    
                    ANN_input(1:15) = thetao_zgrad_sigma/thetao_zgrad_sigma_dist
                    ANN_input(16:30) = PRHO_zgrad_sigma/PRHO_zgrad_sigma_dist
                    
                    ANN_input(31) = (log10(mld_depth) - 1.0) / 2.5
                    ANN_input(32) = sin(pi / 180.0 * ml_data%geoLatT)
                    ANN_input(33) = (log10(tauamp+1E-3)+1.2)/0.46
                    ANN_input(34) = (ml_data%latent+114)/71
                    ANN_input(35) = (ml_data%sensible+14.4)/24
                    ANN_input(36) = (ml_data%lw+55)/21
                    ANN_input(37) = ml_data%sw/400
                    ANN_input(38:52) = shear2_sigma/shear2_sigma_dist

                    ANN_input(53) = (log10(thetao_zgrad_sigma_dist)+0.8)/0.5
                    ANN_input(54) = (log10(PRHO_zgrad_sigma_dist)+1.35)/0.55
                    ANN_input(55) = (log10(shear2_sigma_dist)+4.15)/0.86

                    ANN_input_final(1:7) = ANN_input(31:37)
                    ANN_input_final(8:10) = ANN_input(53:55)
                    
                    call cnn_encode(ANN_input(1:15), ml_config%e1_weight1, ml_config%e1_bias1, ml_config%e1_weight2, ml_config%e1_bias2, encoder_output)
                    ANN_input_final(11:18) = encoder_output
                    call cnn_encode(ANN_input(16:30), ml_config%e2_weight1, ml_config%e2_bias1, ml_config%e2_weight2, ml_config%e2_bias2, encoder_output)
                    ANN_input_final(19:26) = encoder_output
                    call cnn_encode(ANN_input(38:52), ml_config%e3_weight1, ml_config%e3_bias1, ml_config%e3_weight2, ml_config%e3_bias2, encoder_output)
                    ANN_input_final(27:34) = encoder_output

                    attns1 = max(ReLU_zero, matmul(ml_config%attn_weight1, ANN_input_final) + ml_config%attn_bias1)
                    attns = matmul(ml_config%attn_weight2, attns1) + ml_config%attn_bias2
                    do i = 1, 34
                        exp_x(i) = exp(attns(i))
                    end do

                    sum_exp = sum(exp_x)

                    do i = 1, 34
                        attns(i) = exp_x(i) / sum_exp
                    end do

                    ANN_input_final = ANN_input_final*attns

                    l1_output = max(ReLU_zero, matmul(ml_config%l1_weight, ANN_input_final) + ml_config%l1_bias)
                    l2_output = max(ReLU_zero, matmul(ml_config%l2_weight, l1_output) + ml_config%l2_bias)
                    l3_output = matmul(ml_config%l3_weight, l2_output) + ml_config%l3_bias
                    
                    ! l3_output is the latent output

                    !decoder: from l3_output to flux_output
                    d_output1 = max(ReLU_zero, matmul(ml_config%d_weight1, l3_output) + ml_config%d_bias1)
                    idx = 1
                    do i = 1, 16       ! rows
                        do j = 1, 8     ! columns
                            d_output1_reshaped(i, j) = d_output1(idx)
                            idx = idx + 1
                        end do
                    end do
                    call transposed_conv1d(d_output1_reshaped, ml_config%d_weight2, ml_config%d_bias2, decoder_output)
                    do i = 1, 8       ! rows
                        do j = 1, 16     ! columns
                            if (decoder_output(i,j) < 0.0) decoder_output(i,j) = 0.0
                        end do
                    end do
                    ! Final conv1d channels (8 to 1)
                   
                    
                    do j = 1, 16
                        flux_output(j) = ml_config%d_bias3
                        do k = 1, 8
                            do l = -1,1
                            if (j+l >= 1 .and. j+l <= 16) then
                                flux_output(j) = flux_output(j) + ml_config%d_weight3(1,k,l+2)*decoder_output(k,j+l)
                            endif
                            enddo
                        enddo
                    enddo
                
                   

                    coef = thetao_zgrad_sigma_dist*0.1*mld_depth*(tauamp/rho0)**0.5
                    flux_output = flux_output * coef
        
                    output_DT_sigmas =  (flux_output(1:15)-flux_output(2:16))/(0.2*mld_depth)/1000
                    
                    
                
                    allocate(output_DT_at_zl(zl_index_3mld))
                    do zz = 1, zl_index_3mld
                        call find_right_index_clean(target_sigmas, zl_to_sigma(zz), right_index)
                        if (right_index == 0) then
                            output_DT_at_zl(zz) = 0.0
                        else if (right_index == 1) then
                            output_DT_at_zl(zz) = output_DT_sigmas(1)
                        else
                            call interpolate(target_sigmas(right_index-1),target_sigmas(right_index),output_DT_sigmas(right_index-1),&
                                    output_DT_sigmas(right_index),zl_to_sigma(zz),output_DT_at_zl(zz))
                        end if
                    end do

                    ml_data%T_inc(1:zl_index_3mld) = output_DT_at_zl

                    
                    
                    
                    all_mask = (/ ml_data%mask2dT, ml_data%OBCmaskCu_left, ml_data%OBCmaskCu_right, ml_data%OBCmaskCv_south, ml_data%OBCmaskCv_north, &
                               ml_data%mask2dT_left, ml_data%mask2dT_right, ml_data%mask2dT_south, ml_data%mask2dT_north, &
                               ml_data%OBCmaskCu_left_south, ml_data%OBCmaskCu_left_north, ml_data%OBCmaskCu_right_south, ml_data%OBCmaskCu_right_north, &
                               ml_data%OBCmaskCv_south_left, ml_data%OBCmaskCv_south_right, ml_data%OBCmaskCv_north_left, ml_data%OBCmaskCv_north_right /)



                    if (MINVAL(ml_data%all_bathy) > 1110.0 .and. MINVAL(all_mask) > 0.0) then
                        
                        call oda_ml_inference_ann2(ml_config,ml_data)
                        
                    endif



                endif
            endif ! end if 3 mld exceeds total number of levels
        
        endif
        ml_data%S_inc=0.0

    end subroutine oda_ml_inference

    soubroutine oda_ml_inference_ann2(ml_config, ml_data)
        type(ocean_oda_ml_config), pointer, intent(in) :: ml_config
        type(ocean_oda_ml_data), pointer, intent(in) :: ml_data
        
        
        real :: thetao_top, thetao_bottom, so_top,so_bottom, PRHO_top, PRHO_bottom, CT
        real :: thetao_right, thetao_left, so_right,so_left, PRHO_right, PRHO_left
        real :: , uo_right_top, uo_right_bottom, uo_left_top, uo_left_bottom
        real :: vo_north_top, vo_north_bottom, vo_south_top, vo_south_bottom
        real :: uo_zgrad, vo_zgrad, vor_ls, vor_rs, vor_ln, vor_rn
        real :: strain1_ls, strain1_ln, strain1_rs, strain1_rn, strain1, strain2
        real :: thetao, so, PRHO, tauamp, taucurl, taucurl_rs, taucurl_rn, taucurl_ln, taucurl_ls

        real, dimension(50) :: thetao_zgrad, PRHO_zgrad,  vor, div, shear
        real, dimension(50) :: thetao_xgrad, PRHO_xgrad, thetao_ygrad, PRHO_ygrad
        real, dimension(50) :: strain, output_DT_at_zl
        real :: thetao_zgrad_dist, PRHO_zgrad_dist, shear_dist, vor_dist, div_dist
        real :: thetao_xgrad_dist, PRHO_xgrad_dist, thetao_ygrad_dist, PRHO_ygrad_dist, strain_dist
        
        real, dimension(517) :: ANN_input
        real, dimension(97) :: ANN_input_final, exp_x, attns
        real, dimension(8) :: encoder_output
        real, dimension(32) :: l1_output, l2_output, attns1
        real, dimension(8) :: l3_output
        real, dimension(208) :: d_output1
        real, dimension(16,13) :: d_output1_reshaped
        real, dimension(8,25) :: decoder_output1
        real, dimension(4,50) :: decoder_output2
        integer :: zz, i, j, idx, k, l
        real :: sum_exp, pi, ReLU_zero
        real, dimension(:), allocatable :: z_l

        ReLU_zero = 0.0
        pi = acos(-1.0)
        allocate(z_l(ml_config%nk),source=0.0)
        z_l = ml_config%z_l

        
        !prepare for inputs
        do zz = 1, 50
            thetao_top = ml_data%T(zz)
            so_top = ml_data%S(zz)
            CT = gsw_ct_from_pt(so_top,thetao_top)
            PRHO_top = gsw_sigma0(so_top,CT)
            uo_right_top = ml_data%U_right(zz)
            uo_left_top = ml_data%U_left(zz)
            vo_north_top = ml_data%V_north(zz)
            vo_south_top = ml_data%V_south(zz)
                    
            thetao_bottom = ml_data%T(zz+1)
            so_bottom = ml_data%S(zz+1)
            CT = gsw_ct_from_pt(so_bottom,thetao_bottom)
            PRHO_bottom = gsw_sigma0(so_bottom,CT)
            uo_right_bottom = ml_data%U_right(zz+1)
            uo_left_bottom = ml_data%U_left(zz+1)
            vo_north_bottom = ml_data%V_north(zz+1)
            vo_south_bottom = ml_data%V_south(zz+1)

            thetao_zgrad(zz) = (thetao_top - thetao_bottom)/(z_l(zz+1) - z_l(zz))
            PRHO_zgrad(zz) = (PRHO_top - PRHO_bottom)/(z_l(zz+1) - z_l(zz))
            uo_zgrad = (uo_right_top-uo_right_bottom+uo_left_top-uo_left_bottom)/(2*(z_l(zz+1) - z_l(zz)))
            vo_zgrad = (vo_north_top-vo_north_bottom+vo_south_top-vo_south_bottom)/(2*(z_l(zz+1) - z_l(zz)))
            shear(zz) = uo_zgrad **2 + vo_zgrad**2
        end do

        do zz = 1, 50
            call compute_current_divergence(ml_data%U_left(zz)*ml_data%dyCu_left, ml_data%U_right(zz)*ml_data%dyCu_right, &
                                        ml_data%V_south(zz)*ml_data%dxCv_south, ml_data%V_north(zz)*ml_data%dxCv_north, &
                                        ml_data%areacello, div(zz))

            call compute_current_vorticity(ml_data%U_left(zz)*ml_data%dxCu_left, ml_data%U_left_north(zz)*ml_data%dxCu_left_north, &
                                        ml_data%V_north_left(zz)*ml_data%dyCv_north_left, ml_data%V_north(zz)*ml_data%dyCv_north, &
                                        ml_data%areacellobu_left_north, vor_ln)
            call compute_current_vorticity(ml_data%U_right(zz)*ml_data%dxCu_right, ml_data%U_right_north(zz)*ml_data%dxCu_right_north, &
                                        ml_data%V_north(zz)*ml_data%dyCv_north, ml_data%V_north_right(zz)*ml_data%dyCv_north_right, &
                                        ml_data%areacellobu_right_north, vor_rn)
            call compute_current_vorticity(ml_data%U_left_south(zz)*ml_data%dxCu_left_south, ml_data%U_left(zz)*ml_data%dxCu_left, &
                                        ml_data%V_south_left(zz)*ml_data%dyCv_south_left, ml_data%V_south(zz)*ml_data%dyCv_south, &
                                        ml_data%areacellobu_left_south, vor_ls)
            call compute_current_vorticity(ml_data%U_right_south(zz)*ml_data%dxCu_right_south, ml_data%U_right(zz)*ml_data%dxCu_right, &
                                        ml_data%V_south(zz)*ml_data%dyCv_south, ml_data%V_south_right(zz)*ml_data%dyCv_south_right, &
                                        ml_data%areacellobu_right_south, vor_rs)
            vor(zz) = (vor_ln+vor_rn+vor_ls+vor_rs)/4
            call compute_current_strain1(ml_data%U_left(zz)*ml_data%dxCu_left, ml_data%U_left_north(zz)*ml_data%dxCu_left_north, &
                                        ml_data%V_north_left(zz)*ml_data%dyCv_north_left, ml_data%V_north(zz)*ml_data%dyCv_north, &
                                        ml_data%areacellobu_left_north, strain1_ln)
            call compute_current_strain1(ml_data%U_right(zz)*ml_data%dxCu_right, ml_data%U_right_north(zz)*ml_data%dxCu_right_north, &
                                        ml_data%V_north(zz)*ml_data%dyCv_north, ml_data%V_north_right(zz)*ml_data%dyCv_north_right, &
                                        ml_data%areacellobu_right_north, strain1_rn)
            call compute_current_strain1(ml_data%U_left_south(zz)*ml_data%dxCu_left_south, ml_data%U_left(zz)*ml_data%dxCu_left, &
                                        ml_data%V_south_left(zz)*ml_data%dyCv_south_left, ml_data%V_south(zz)*ml_data%dyCv_south, &
                                        ml_data%areacellobu_left_south, strain1_ls)
            call compute_current_strain1(ml_data%U_right_south(zz)*ml_data%dxCu_right_south, ml_data%U_right(zz)*ml_data%dxCu_right, &
                                        ml_data%V_south(zz)*ml_data%dyCv_south, ml_data%V_south_right(zz)*ml_data%dyCv_south_right, &
                                        ml_data%areacellobu_right_south, strain1_rs)
            call compute_current_strain2(ml_data%U_left(zz)*ml_data%dyCu_left, ml_data%U_right(zz)*ml_data%dyCu_right, &
                                        ml_data%V_south(zz)*ml_data%dxCv_south, ml_data%V_north(zz)*ml_data%dxCv_north, &
                                        ml_data%areacello, strain2)
            strain1 = (strain1_ln+strain1_rn+strain1_ls+strain1_rs)/4
            strain(zz) = (strain1**2+strain2**2)**0.5

            thetao_left = ml_data%T_left(zz)
            so_left = ml_data%S_left(zz)
            CT = gsw_ct_from_pt(so_left,thetao_left)
            PRHO_left = gsw_sigma0(so_left,CT)

            thetao_right = ml_data%T_right(zz)
            so_right = ml_data%S_right(zz)
            CT = gsw_ct_from_pt(so_right,thetao_right)
            PRHO_right = gsw_sigma0(so_right,CT)

            thetao = ml_data%T(zz)
            so = ml_data%S(zz)
            CT = gsw_ct_from_pt(so,thetao)
            PRHO = gsw_sigma0(so,CT)

            thetao_xgrad(zz) = ((thetao_right-thetao)/ml_data%dxCu_right + (thetao-thetao_left)/ml_data%dxCu_left)/2
            PRHO_xgrad(zz) = ((PRHO_right-PRHO)/ml_data%dxCu_right + (PRHO-PRHO_left)/ml_data%dxCu_left)/2

            thetao_north = ml_data%T_north(zz)
            so_north = ml_data%S_north(zz)
            CT = gsw_ct_from_pt(so_north,thetao_north)
            PRHO_north = gsw_sigma0(so_north,CT)

            thetao_south = ml_data%T_south(zz)
            so_south = ml_data%S_south(zz)
            CT = gsw_ct_from_pt(so_south,thetao_south)
            PRHO_south = gsw_sigma0(so_south,CT)


            thetao_ygrad(zz) = ((thetao_north-thetao)/ml_data%dyCv_north + (thetao-thetao_south)/ml_data%dyCv_south)/2
            PRHO_ygrad(zz) = ((PRHO_north-PRHO)/ml_data%dyCv_north + (PRHO-PRHO_south)/ml_data%dyCv_south)/2

        end do

        tauamp = sqrt(((ml_data%taux_left+ml_data%taux_right)/2)**2+((ml_data%tauy_south+ml_data%tauy_north)/2)**2)
        call compute_current_vorticity(ml_data%taux_left*ml_data%dxCu_left, ml_data%taux_left_north*ml_data%dxCu_left_north, &
                                        ml_data%tauy_north_left*ml_data%dyCv_north_left, ml_data%tauy_north*ml_data%dyCv_north, &
                                        ml_data%areacellobu_left_north, taucurl_ln)
        call compute_current_vorticity(ml_data%taux_right*ml_data%dxCu_right, ml_data%taux_right_north*ml_data%dxCu_right_north, &
                                        ml_data%tauy_north*ml_data%dyCv_north, ml_data%tauy_north_right*ml_data%dyCv_north_right, &
                                        ml_data%areacellobu_right_north, taucurl_rn)
        call compute_current_vorticity(ml_data%taux_left_south*ml_data%dxCu_left_south, ml_data%taux_left*ml_data%dxCu_left, &
                                        ml_data%tauy_south_left*ml_data%dyCv_south_left, ml_data%tauy_south*ml_data%dyCv_south, &
                                        ml_data%areacellobu_left_south, taucurl_ls)
        call compute_current_vorticity(ml_data%taux_right_south*ml_data%dxCu_right_south, ml_data%taux_right*ml_data%dxCu_right, &
                                        ml_data%tauy_south*ml_data%dyCv_south, ml_data%tauy_south_right*ml_data%dyCv_south_right, &
                                        ml_data%areacellobu_right_south, taucurl_rs)
        taucurl = (taucurl_ln+taucurl_rn+taucurl_ls+taucurl_rs)/4

        thetao_zgrad_dist = sqrt(sum(thetao_zgrad**2))
        PRHO_zgrad_dist = sqrt(sum(PRHO_zgrad**2))
        shear_dist = sqrt(sum(shear**2))
        vor_dist = sqrt(sum(vor**2))
        div_dist = sqrt(sum(div**2))
        thetao_xgrad_dist = sqrt(sum(thetao_xgrad**2))
        thetao_ygrad_dist = sqrt(sum(thetao_ygrad**2))
        PRHO_xgrad_dist = sqrt(sum(PRHO_xgrad**2))
        PRHO_ygrad_dist = sqrt(sum(PRHO_ygrad**2))
        strain_dist = sqrt(sum(strain**2))
        
                    
        ANN_input(1:50) = thetao_zgrad/thetao_zgrad_dist
        ANN_input(51:100) = PRHO_zgrad/PRHO_zgrad_dist
        ANN_input(101:150) = shear/shear_dist
        ANN_input(151:200) = vor/vor_dist
        ANN_input(201:250) = div/div_dist
                    
        ANN_input(251) = sin(pi / 180.0 * ml_data%geoLatT)
        ANN_input(252) = (log10(tauamp+1E-3)+1.2)/0.46
        ANN_input(253) = taucurl*1E5
        ANN_input(254) = (ml_data%latent+114)/71
        ANN_input(255) = (ml_data%sensible+14.4)/24
        ANN_input(256) = (ml_data%lw+55)/21
        ANN_input(257) = ml_data%sw/400

        ANN_input(258:307) = thetao_xgrad/thetao_xgrad_dist
        ANN_input(308:357) = thetao_ygrad/thetao_ygrad_dist
        ANN_input(358:407) = PRHO_xgrad/PRHO_xgrad_dist
        ANN_input(408:457) = PRHO_ygrad/PRHO_ygrad_dist
        ANN_input(458:507) = strain/strain_dist
        
        ANN_input(508) = (log10(thetao_zgrad_dist+1E-2)+0.53)/0.4
        ANN_input(509) = (log10(PRHO_zgrad_dist+1E-3)+1.15)/0.51
        ANN_input(510) = (log10(shear_dist+1E-6)+3.47)/0.56
        ANN_input(511) = (log10(vor_dist)+5.5)/0.44
        ANN_input(512) = (log10(div_dist)+5.75)/0.41
        ANN_input(513) = (log10(thetao_xgrad_dist+1E-7)+4.74)/0.3
        ANN_input(514) = (log10(thetao_ygrad_dist+1E-6)+4.43)/0.26
        ANN_input(515) = (log10(PRHO_xgrad_dist+1E-7)+5.37)/0.3
        ANN_input(516) = (log10(PRHO_ygrad_dist)+5.1)/0.34
        ANN_input(517) = (log10(strain_dist)+5.35)/0.39
        


        ANN_input(53) = (log10(thetao_zgrad_sigma_dist)+0.8)/0.5
        ANN_input(54) = (log10(PRHO_zgrad_sigma_dist)+1.35)/0.55
        ANN_input(55) = (log10(shear2_sigma_dist)+4.15)/0.86

        ANN_input_final(1:7) = ANN_input(251:257)
        ANN_input_final(8:17) = ANN_input(508:517)

        call cnn_encode_ann2(ANN_input(1:50), ml_config%e1_weight1_ann2, ml_config%e1_bias1_ann2, ml_config%e1_weight2_ann2, ml_config%e1_bias2_ann2, encoder_output)
        ANN_input_final(18:25) = encoder_output
        call cnn_encode_ann2(ANN_input(51:100), ml_config%e2_weight1_ann2, ml_config%e2_bias1_ann2, ml_config%e2_weight2_ann2, ml_config%e2_bias2_ann2, encoder_output)
        ANN_input_final(26:33) = encoder_output
        call cnn_encode_ann2(ANN_input(101:150), ml_config%e3_weight1_ann2, ml_config%e3_bias1_ann2, ml_config%e3_weight2_ann2, ml_config%e3_bias2_ann2, encoder_output)
        ANN_input_final(34:41) = encoder_output
        call cnn_encode_ann2(ANN_input(151:200), ml_config%e4_weight1_ann2, ml_config%e4_bias1_ann2, ml_config%e4_weight2_ann2, ml_config%e4_bias2_ann2, encoder_output)
        ANN_input_final(42:49) = encoder_output
        call cnn_encode_ann2(ANN_input(201:250), ml_config%e5_weight1_ann2, ml_config%e5_bias1_ann2, ml_config%e5_weight2_ann2, ml_config%e5_bias2_ann2, encoder_output)
        ANN_input_final(50:57) = encoder_output
        call cnn_encode_ann2(ANN_input(258:307), ml_config%e6_weight1_ann2, ml_config%e6_bias1_ann2, ml_config%e6_weight2_ann2, ml_config%e6_bias2_ann2, encoder_output)
        ANN_input_final(58:65) = encoder_output
        call cnn_encode_ann2(ANN_input(308:357), ml_config%e7_weight1_ann2, ml_config%e7_bias1_ann2, ml_config%e7_weight2_ann2, ml_config%e7_bias2_ann2, encoder_output)
        ANN_input_final(66:73) = encoder_output
        call cnn_encode_ann2(ANN_input(358:407), ml_config%e8_weight1_ann2, ml_config%e8_bias1_ann2, ml_config%e8_weight2_ann2, ml_config%e8_bias2_ann2, encoder_output)
        ANN_input_final(74:81) = encoder_output
        call cnn_encode_ann2(ANN_input(408:457), ml_config%e9_weight1_ann2, ml_config%e9_bias1_ann2, ml_config%e9_weight2_ann2, ml_config%e9_bias2_ann2, encoder_output)
        ANN_input_final(82:89) = encoder_output
        call cnn_encode_ann2(ANN_input(458:507), ml_config%e10_weight1_ann2, ml_config%e10_bias1_ann2, ml_config%e10_weight2_ann2, ml_config%e10_bias2_ann2, encoder_output)
        ANN_input_final(90:97) = encoder_output

        attns1 = max(ReLU_zero, matmul(ml_config%attn_weight1_ann2, ANN_input_final) + ml_config%attn_bias1_ann2)
        attns = matmul(ml_config%attn_weight2_ann2, attns1) + ml_config%attn_bias2_ann2
        do i = 1, 97
            exp_x(i) = exp(attns(i))
        end do

        sum_exp = sum(exp_x)

        do i = 1, 97
            attns(i) = exp_x(i) / sum_exp
        end do

        ANN_input_final = ANN_input_final*attns

        l1_output = max(ReLU_zero, matmul(ml_config%l1_weight_ann2, ANN_input_final) + ml_config%l1_bias_ann2)
        l2_output = max(ReLU_zero, matmul(ml_config%l2_weight_ann2, l1_output) + ml_config%l2_bias_ann2)
        l3_output = matmul(ml_config%l3_weight_ann2, l2_output) + ml_config%l3_bias_ann2
                    
        ! l3_output is the latent output

        !decoder: from l3_output to DT_at_zl
        d_output1 = max(ReLU_zero, matmul(ml_config%d_weight1_ann2, l3_output) + ml_config%d_bias1_ann2)
        idx = 1
        do i = 1, 16       ! rows
            do j = 1, 13     ! columns
                d_output1_reshaped(i, j) = d_output1(idx)
                idx = idx + 1
            end do
        end do
        call transposed_conv1d_ann2_1(d_output1_reshaped, ml_config%d_weight2_ann2, ml_config%d_bias2_ann2, decoder_output1)
        do i = 1, 8       ! rows
            do j = 1, 25     ! columns
                if (decoder_output1(i,j) < 0.0) decoder_output1(i,j) = 0.0
            end do
        end do

        call transposed_conv1d_ann2_2(decoder_output1, ml_config%d_weight3_ann2, ml_config%d_bias3_ann2, decoder_output2)
        do i = 1, 4       ! rows
            do j = 1, 50     ! columns
                if (decoder_output2(i,j) < 0.0) decoder_output2(i,j) = 0.0
            end do
        end do

        ! Final conv1d channels (4 to 1)
        !output_DT_at_zl
        do j = 1, 50
            output_DT_at_zl(j) = ml_config%d_bias4_ann2
            do k = 1, 4
                do l = -1,1
                if (j+l >= 1 .and. j+l <= 50) then
                    output_DT_at_zl(j) = output_DT_at_zl(j) + ml_config%d_weight4_ann2(1,k,l+2)*decoder_output2(k,j+l)
                endif
                enddo
            enddo
        enddo

        

        ml_data%T_inc(1:50) = ml_data%T_inc(1:50) + output_DT_at_zl/seconds_in_30_days
                    
        
        
    end soubroutine oda_ml_inference_ann2

    subroutine oda_ml_init(ml_config,ml_data,GV)
        type(ocean_oda_ml_config), pointer, intent(in) :: ml_config
        type(ocean_oda_ml_data), pointer, intent(in) :: ml_data
        type(verticalGrid_type), pointer, intent(in) :: GV   !< The ocean's vertical grid structure

        ! load the NN weights and biases
        call read_ANN_file(ml_config)

        allocate(ml_config%z_l(GV%ke), source=0.0)
        ml_config%z_l = GV%sLayer

        allocate(ml_config%z_i(GV%ke+1), source=0.0)
        ml_config%z_i = GV%sInterface

        ml_config%nk = GV%ke
        call init_oda_ml_features(ml_data,GV%ke)    

    end subroutine oda_ml_init

    subroutine oda_ml_end(ml_config,ml_data)
        type(ocean_oda_ml_config), pointer, intent(in) :: ml_config
        type(ocean_oda_ml_data), pointer, intent(in) :: ml_data

    end subroutine oda_ml_end

    subroutine init_oda_ml_features(ml_data,nk)
        type(ocean_oda_ml_data), pointer, intent(in) :: ml_data
        integer, intent(in) :: nk

        ml_data%nk = nk
        allocate(ml_data%T(nk),source=0.0)
        allocate(ml_data%S(nk),source=0.0)
        allocate(ml_data%U_left(nk),source=0.0)
        allocate(ml_data%U_right(nk),source=0.0)
        allocate(ml_data%V_north(nk),source=0.0)
        allocate(ml_data%V_south(nk),source=0.0)

        allocate(ml_data%T_inc(nk),source=0.0)
        allocate(ml_data%S_inc(nk),source=0.0)

    end subroutine init_oda_ml_features

    subroutine cnn_encode(input_vec, weights1, bias1, weights2, bias2, output_vec)
        implicit none

        real, dimension(15), intent(in)  :: input_vec
        real, dimension(16), intent(in)  :: bias1
        real, dimension(8), intent(in)  :: bias2
        real, dimension(16, 1, 3), intent(in)  :: weights1
        real, dimension(8, 16, 3), intent(in)  :: weights2
        real, dimension(8), intent(out) :: output_vec

        real, dimension(16, 15) :: layer1_output
        real, dimension(8, 15) :: layer2_output
        integer :: i, j, k, l

        ! First conv layer: 1 → 16 channels
        do i = 1, 16
          do j = 1, 15
            layer1_output(i,j) = bias1(i)
            do k = -1,1
              if (j+k >= 1 .and. j+k <= 15) then
                layer1_output(i,j) = layer1_output(i,j) + weights1(i,1,k+2)*input_vec(j+k)
              endif
            enddo
            if (layer1_output(i,j) < 0.0) layer1_output(i,j) = 0.0  ! ReLU
          enddo
        enddo

        ! Second conv layer: 16 → 8 channels
        do i = 1, 8
          do j = 1, 15
            layer2_output(i,j) = bias2(i)
            do k = 1, 16
              do l = -1,1
                if (j+l >= 1 .and. j+l <= 15) then
                  layer2_output(i,j) = layer2_output(i,j) + weights2(i,k,l+2)*layer1_output(k,j+l)
                endif
              enddo
            enddo
            if (layer2_output(i,j) < 0.0) layer2_output(i,j) = 0.0  ! ReLU
          enddo
        enddo

        ! Adaptive avg pooling: avg over 15 timesteps → (8)
        do i = 1, 8
          output_vec(i) = sum(layer2_output(i,1:15)) / 15.0
        enddo
      end subroutine cnn_encode

    subroutine cnn_encode_ann2(input_vec, weights1, bias1, weights2, bias2, output_vec)
        implicit none

        real, dimension(50), intent(in)  :: input_vec
        real, dimension(16), intent(in)  :: bias1
        real, dimension(8), intent(in)  :: bias2
        real, dimension(16, 1, 3), intent(in)  :: weights1
        real, dimension(8, 16, 3), intent(in)  :: weights2
        real, dimension(8), intent(out) :: output_vec

        real, dimension(16, 50) :: layer1_output
        real, dimension(8, 50) :: layer2_output
        integer :: i, j, k, l

        ! First conv layer: 1 → 16 channels
        do i = 1, 16
          do j = 1, 50
            layer1_output(i,j) = bias1(i)
            do k = -1,1
              if (j+k >= 1 .and. j+k <= 50) then
                layer1_output(i,j) = layer1_output(i,j) + weights1(i,1,k+2)*input_vec(j+k)
              endif
            enddo
            if (layer1_output(i,j) < 0.0) layer1_output(i,j) = 0.0  ! ReLU
          enddo
        enddo

        ! Second conv layer: 16 → 8 channels
        do i = 1, 8
          do j = 1, 50
            layer2_output(i,j) = bias2(i)
            do k = 1, 16
              do l = -1,1
                if (j+l >= 1 .and. j+l <= 50) then
                  layer2_output(i,j) = layer2_output(i,j) + weights2(i,k,l+2)*layer1_output(k,j+l)
                endif
              enddo
            enddo
            if (layer2_output(i,j) < 0.0) layer2_output(i,j) = 0.0  ! ReLU
          enddo
        enddo

        ! Adaptive avg pooling: avg over 15 timesteps → (8)
        do i = 1, 8
          output_vec(i) = sum(layer2_output(i,1:50)) / 50.0
        enddo
      end subroutine cnn_encode

    subroutine transposed_conv1d(input, weight, bias, output)
      real, dimension(16,8), intent(in) :: input             ! (in_channels, length)
      real, dimension(16,8,3), intent(in) :: weight          ! (in_channels, out_channels, kernel)
      real, dimension(8), intent(in) :: bias
      real, dimension(8,16), intent(out) :: output           ! (out_channels, output_length)

      integer :: in_ch, out_ch, k, t, out_pos

      output = 0.0

      do in_ch = 1, 16
         do t = 1, 8
            do out_ch = 1, 8
               do k = 1, 3
                  out_pos = (t - 1)*2 - 1 + k ! 2: stride; 1: padding; k: kernel
                  if (out_pos >= 1 .and. out_pos <= 16) then
                     output(out_ch, out_pos) = output(out_ch, out_pos) + &
                          input(in_ch, t) * weight(in_ch, out_ch, k)
                  end if
               end do
            end do
         end do
      end do

      ! Add bias
      do out_ch = 1, 8
         output(out_ch, :) = output(out_ch, :) + bias(out_ch)
      end do

    end subroutine transposed_conv1d

    subroutine transposed_conv1d_ann2_1(input, weight, bias, output)
      real, dimension(16,13), intent(in) :: input             ! (in_channels, length)
      real, dimension(16,8,3), intent(in) :: weight          ! (in_channels, out_channels, kernel)
      real, dimension(8), intent(in) :: bias
      real, dimension(8,25), intent(out) :: output           ! (out_channels, output_length)

      integer :: in_ch, out_ch, k, t, out_pos

      output = 0.0

      do in_ch = 1, 16
         do t = 1, 13
            do out_ch = 1, 8
               do k = 1, 3
                  out_pos = (t - 1)*2 - 1 + k -1 ! 2: stride; 1: padding; k: kernel (k-1), +out_padding = 0
                  if (out_pos >= 1 .and. out_pos <= 25) then
                     output(out_ch, out_pos) = output(out_ch, out_pos) + &
                          input(in_ch, t) * weight(in_ch, out_ch, k)
                  end if
               end do
            end do
         end do
      end do

      ! Add bias
      do out_ch = 1, 8
         output(out_ch, :) = output(out_ch, :) + bias(out_ch)
      end do

    end subroutine transposed_conv1d_ann2_1


    subroutine transposed_conv1d_ann2_2(input, weight, bias, output)
      real, dimension(8,25), intent(in) :: input             ! (in_channels, length)
      real, dimension(8,4,3), intent(in) :: weight          ! (in_channels, out_channels, kernel)
      real, dimension(4), intent(in) :: bias
      real, dimension(4,50), intent(out) :: output           ! (out_channels, output_length)

      integer :: in_ch, out_ch, k, t, out_pos

      output = 0.0

      do in_ch = 1, 8
         do t = 1, 25
            do out_ch = 1, 4
               do k = 1, 3
                  out_pos = (t - 1)*2 - 1 + k ! 2: stride; 1: padding; k: kernel
                  if (out_pos >= 1 .and. out_pos <= 50) then
                     output(out_ch, out_pos) = output(out_ch, out_pos) + &
                          input(in_ch, t) * weight(in_ch, out_ch, k)
                  end if
               end do
            end do
         end do
      end do

      ! Add bias
      do out_ch = 1, 4
         output(out_ch, :) = output(out_ch, :) + bias(out_ch)
      end do

    end subroutine transposed_conv1d_ann2_2


    Subroutine read_ANN_file(ml_config)
        implicit none
        type(ocean_oda_ml_config), pointer, intent(in) :: ml_config

        
        real, dimension(3,1,16)  :: e1_weight1_temp, e2_weight1_temp, e3_weight1_temp
        real, dimension(3,16,8)  :: e1_weight2_temp, e2_weight2_temp, e3_weight2_temp
        real, dimension(34,32)  :: attn_weight1_temp
        real, dimension(32,34)  :: attn_weight2_temp
        real, dimension(34,32)  :: l1_weight_temp
        real, dimension(32,32) :: l2_weight_temp
        real, dimension(32,8) :: l3_weight_temp
        real, dimension(8,128) :: d_weight1_temp
        real, dimension(3,8,16)  :: d_weight2_temp
        real, dimension(3,8,1)  :: d_weight3_temp


        integer :: ncid, varid, retval, i, j, k
        character(len = 255) :: varname

        ml_config%filename = danni_ANN_name

        ! Open the NetCDF file
        retval = nf90_open(ml_config%filename, nf90_nowrite, ncid)
        if (retval /= nf90_noerr) then
        print *, 'Error: Unable to open file'
        stop
        endif

        ! Get the variable ID
        varname = 'e1_weight1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, e1_weight1_temp)
        do i = 1, 3
            do k = 1, 16
                ml_config%e1_weight1(k, 1, i) = e1_weight1_temp(i, 1, k)
            end do
       end do

        varname = 'e2_weight1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, e2_weight1_temp)
        do i = 1, 3
            do k = 1, 16
                ml_config%e2_weight1(k, 1, i) = e2_weight1_temp(i, 1, k)
            end do
        end do

        varname = 'e3_weight1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, e3_weight1_temp)
        do i = 1, 3
            do k = 1, 16
                ml_config%e3_weight1(k, 1, i) = e3_weight1_temp(i, 1, k)
            end do
        end do

        varname = 'e1_weight2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, e1_weight2_temp)
        do i = 1, 3
            do j = 1, 16
                do k = 1, 8
                    ml_config%e1_weight2(k, j, i) = e1_weight2_temp(i, j, k)
                end do
            end do
        end do

        varname = 'e2_weight2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, e2_weight2_temp)
        do i = 1, 3
            do j = 1, 16
                do k = 1, 8
                    ml_config%e2_weight2(k, j, i) = e2_weight2_temp(i, j, k)
                end do
            end do
        end do

        varname = 'e3_weight2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, e3_weight2_temp)
        do i = 1, 3
            do j = 1, 16
                do k = 1, 8
                    ml_config%e3_weight2(k, j, i) = e3_weight2_temp(i, j, k)
                end do
            end do
        end do

        varname = 'd_weight2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, d_weight2_temp)
        do i = 1, 3
            do j = 1, 8
                do k = 1, 16
                    ml_config%d_weight2(k, j, i) = d_weight2_temp(i, j, k)
                end do
            end do
        end do

        varname = 'd_weight3'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, d_weight3_temp)
        do i = 1, 3
            do j = 1, 8
               ml_config%d_weight3(1, j, i) = d_weight3_temp(i, j, 1)
            end do
        end do
        


        varname = 'attn_weight1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, attn_weight1_temp)
        ml_config%attn_weight1 = transpose(attn_weight1_temp)

        varname = 'attn_weight2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, attn_weight2_temp)
        ml_config%attn_weight2 = transpose(attn_weight2_temp)

        varname = 'l1_weight'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, l1_weight_temp)
        ml_config%l1_weight = transpose(l1_weight_temp)

        varname = 'l2_weight'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, l2_weight_temp)
        ml_config%l2_weight = transpose(l2_weight_temp)
        
        varname = 'l3_weight'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, l3_weight_temp)
        ml_config%l3_weight = transpose(l3_weight_temp)

        varname = 'd_weight1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, d_weight1_temp)
        ml_config%d_weight1 = transpose(d_weight1_temp)
        

        varname = 'l1_bias'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%l1_bias)
        
        varname = 'l2_bias'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%l2_bias)
        
        varname = 'l3_bias'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%l3_bias)

        varname = 'e1_bias1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%e1_bias1)
        varname = 'e1_bias2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%e1_bias2)

        varname = 'e2_bias1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%e2_bias1)
        varname = 'e2_bias2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%e2_bias2)

        varname = 'e3_bias1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%e3_bias1)
        varname = 'e3_bias2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%e3_bias2)

        varname = 'attn_bias1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%attn_bias1)

        varname = 'attn_bias2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%attn_bias2)

        varname = 'd_bias1'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%d_bias1)

        varname = 'd_bias2'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%d_bias2)

        varname = 'd_bias3'
        retval = nf90_inq_varid(ncid, varname, varid)
        retval = nf90_get_var(ncid, varid, ml_config%d_bias3)

        

        ! Close the NetCDF file
        retval = nf90_close(ncid)
        if (retval /= nf90_noerr) then
        print *, 'Error: Unable to close file'
        stop
        endif
    end subroutine read_ANN_file

    ! 1D linear interpolation; it is guaranteed that x1 <= thisx < x2
    subroutine interpolate(x1,x2,y1,y2,thisx,thisy)
        implicit none
        real, intent(in) :: x1, x2, y1, y2, thisx
        real, intent(out) :: thisy
        thisy = (thisx-x1)/(x2-x1)*(y2-y1) + y1
    end subroutine interpolate


    ! Subroutine to find the index. 1D array (index) is greater than the given value
    Subroutine find_right_index(array1d_for_indexing, value_for_indexing, bathy_for_control, array1d_for_depth, right_index)
        implicit none
        real, intent(in) :: array1d_for_indexing(:), array1d_for_depth(:)
        integer :: array1d_i
        integer, intent(out) :: right_index
        real, intent(in) :: value_for_indexing, bathy_for_control
        right_index = 0 ! if there's so such right index, it will be 0 
        do array1d_i = 1, size(array1d_for_indexing)
            if (array1d_for_depth(array1d_i) > bathy_for_control) then
                return
            else if (array1d_for_indexing(array1d_i) > value_for_indexing) then
                right_index = array1d_i
                return
            end if
        end do
    end subroutine find_right_index

    ! Subroutine to find the index. 1D array (index) is greater than the given value, no control value needed.
    Subroutine find_right_index_clean(array1d_for_indexing, value_for_indexing, right_index)
        implicit none
        real(8), intent(in) :: array1d_for_indexing(:)
        integer :: array1d_i
        integer, intent(out) :: right_index
        real(8), intent(in) :: value_for_indexing
        right_index = 0 ! if there's so such right index, it will be 0 
        do array1d_i = 1, size(array1d_for_indexing)
            if (array1d_for_indexing(array1d_i) > value_for_indexing) then
                right_index = array1d_i
                return
            end if
        end do
    end subroutine find_right_index_clean

    ! Subroutine to compute divergence
    Subroutine compute_current_divergence(uy_left, uy_right, vx_south, vx_north, area, this_div)
        implicit none
        real, intent(in) :: uy_left, uy_right, vx_south, vx_north, area
        real, intent(out) :: this_div

        this_div = (uy_right - uy_left + vx_north - vx_south) / area

    end subroutine compute_current_divergence

    Subroutine compute_current_vorticity(ux_south, ux_north, vy_left, vy_right, area, this_vor)
        implicit none
        real, intent(in) :: ux_south, ux_north, vy_left, vy_right, area
        real, intent(out) :: this_vor

        this_vor = (-ux_north + ux_south + vy_right - vy_left) / area

    end subroutine compute_current_vorticity

    Subroutine compute_current_strain1(ux_south, ux_north, vy_left, vy_right, area, this_strain1)
        implicit none
        real, intent(in) :: ux_south, ux_north, vy_left, vy_right, area
        real, intent(out) :: this_strain1

        this_strain1 = (ux_north - ux_south + vy_right - vy_left) / area

    end subroutine compute_current_strain1

    Subroutine compute_current_strain2(uy_left, uy_right, vx_south, vx_north, area, this_strain2)
        implicit none
        real, intent(in) :: uy_left, uy_right, vx_south, vx_north, area
        real, intent(out) :: this_strain2

        this_strain2 = (uy_right - uy_left - vx_north + vx_south) / area

    end subroutine compute_current_strain2

end module MOM_oda_ml_mod
