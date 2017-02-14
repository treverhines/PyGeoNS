import numpy as np
from matplotlib.patches import Polygon
from matplotlib.container import Container

def strain_glyph(x,strain,sigma=None,
                 ext_color='b',cmp_color='r',
                 alpha=0.2,linewidth=1.0,vert=500,
                 scale=1.0,snr_mask=True):
    ''' 
    Returns a container of artists making up a strain glyph.
    
    Parameters
    ----------
    x : (2,) array
      Coordinates of the strain glyph.
    
    strain : (3,) array
      Components of the strain tensor specified as [e_xx,e_yy,_exy].

    sigma : (3,) array or (3,3) array, optional      
      Uncertainty on the strain components. This can either be the 
      standard deviation uncertainty on the components specified as a 
      (3,) array or it can be the covariance of the strain components 
      specified as a (3,3) array.

    ext_color : str, optional
      Extensional color

    cmp_color : str, optional
      Compressional color
    
    alpha : float, optional
      Transparency of the uncertainty field
    
    linewidth : float, optional
      Thickness of the mean lines
    
    vert : int, optional
      Number of vertices used in the strain glyph. Higher values 
      produce a higher resolution glyph at the expense of 
      computational cost.

    scale : float, optional
      Scales the strain and uncertainty. 
    
    snr_mask : bool, optional
      If True, then the strain glyph transparency will be determined 
      by the signal-to-noise ratio (SNR). In this case, the SNR is the 
      ratio of the strain magnitude to its uncertainty. The strain 
      magnitude is the inner product of the strain matrix with itself. 
      If the SNR is less than 1.0, then nothing will be returned. The 
      glyph will be increasingly opaque for SNR ratios between 1.0 and 
      2.0. If the SNR is greater than 2.0 then the expected value 
      lines will be opaque and the uncertainty field will be as 
      specified with *alpha*.

    Returns
    -------
    out : Container instance
      Contains the artists in the strain glyph. Use the *get_label* 
      method to see which component of the strain glyph each artist 
      describes.
    
    Examples
    --------
    Plot a strain glyph which represents extension in the x direction 
    and contraction in the y direction.
    
    >>> x = [0.5,0.5]
    >>> strain = [0.25,-0.25,0.0]
    >>> uncertainty = [0.1,0.1,0.1]
    >>> c = strain_glyph(x,strain,uncertainty)
    >>> fig,ax = plt.subplots()
    >>> for i in c: ax.add_artist(i)
    >>> plt.show()
    
    '''
    x = np.asarray(x,dtype=float)
    strain = np.asarray(strain,dtype=float)
    if sigma is None:
      sigma = np.zeros((3,3))
    else:
      sigma = np.asarray(sigma,dtype=float)

    if sigma.shape == (3,3):
      # if sigma was specified as a (3,3) covariance matrix then store 
      # sigma as is
      cov = sigma
    elif sigma.shape == (3,):
      # if sigma was specified as a (3,) standard deviation 
      # uncertainty vector then convert it to a covariance matrix
      cov = np.diag(sigma**2)
    else:
      raise ValueError('*sigma* must be a (3,) or (3,3) array')
      
    # if either strain or cov are not finite then silently return an 
    # empty container
    if ~np.all(np.isfinite(strain)) | ~np.all(np.isfinite(cov)):
      return Container([])
    
    if snr_mask:
      # strain magnitude # (3,) array
      mag = np.sqrt(strain[0]**2 + strain[1]**2 + 2*strain[2]**2)
      # Jacobian of strain magnitude. used for error propagation 
      jac = np.array([strain[0]/mag,strain[1]/mag,2*strain[2]/mag])
      mag_sigma = np.sqrt(jac.dot(cov).dot(jac))
      snr = mag/mag_sigma
      if snr < 1.0:
        # return no glyph if snr is less than 1.0
        return Container([])
      elif (snr >= 1.0) & (snr < 2.0):
        # return faded glyph if snr is between 1.0 and 2.0
        snr_alpha = snr - 1.0
      else:
        # do not fade if snr > 2,0
        snr_alpha = 1.0
    else:
      # if snr_mask is False, then dont change glyph transparency
      snr_alpha = 1.0
    
    # scale the data
    strain = scale*strain
    cov = scale**2*cov
    # angles for each normal vector
    theta = np.linspace(0.0,2*np.pi,vert)
    # normal vectors
    n = np.array([np.cos(theta),np.sin(theta)]).T
    # This matrix maps the flattened strain tensor to normal strain 
    # along each direction in *n*
    G = np.array([n[:,0]**2, n[:,1]**2,2*n[:,0]*n[:,1]]).T
    mean = G.dot(strain)
    # just compute the diagonals of the covariance matrix
    sigma = np.sqrt(np.sum(G.dot(cov)*G,axis=1))

    artists = []
    # make vertices for the line indicating the expected strain  
    mean_vert = n*mean[:,None]
    # vertices associated with extension
    mean_vert_ext = mean_vert[mean>=0.0]
    # vertices associated with compression
    mean_vert_cmp = mean_vert[mean<0.0]
    # make vertices for the upper bound of strain
    ub_vert = n*(mean[:,None] + sigma[:,None])
    ub_vert_ext = ub_vert[(mean + sigma)>=0.0]
    ub_vert_cmp = ub_vert[(mean + sigma)<0.0]
    # make vertices for the lower bound of strain
    lb_vert = n*(mean[:,None] - sigma[:,None])
    lb_vert_ext = lb_vert[(mean - sigma)>=0.0]
    lb_vert_cmp = lb_vert[(mean - sigma)<0.0]
    # make the vertices defining the 1-sigma extension field
    sigma_vert_ext = np.vstack((ub_vert_ext,lb_vert_ext[::-1]))
    # make the vertices defining the 1-sigma compression field
    sigma_vert_cmp = np.vstack((ub_vert_cmp,lb_vert_cmp[::-1]))
    if mean_vert_ext.shape[0] != 0:
      artists += [Polygon(x + mean_vert_ext,
                          edgecolor=ext_color,
                          facecolor='none',
                          linewidth=linewidth,
                          alpha=snr_alpha,
                          label='extensional mean')]

    if mean_vert_cmp.shape[0] != 0:
      artists += [Polygon(x + mean_vert_cmp,
                          edgecolor=cmp_color,
                          facecolor='none',
                          linewidth=linewidth,
                          alpha=snr_alpha,
                          label='compressional mean')]

    if sigma_vert_ext.shape[0] != 0:
      artists += [Polygon(x + sigma_vert_ext,
                          facecolor=ext_color,
                          edgecolor='none',
                          alpha=alpha*snr_alpha,
                          linewidth=linewidth,
                          label='extensional confidence interval')]

    if sigma_vert_cmp.shape[0] != 0:
      artists += [Polygon(x + sigma_vert_cmp,
                          facecolor=cmp_color,
                          edgecolor='none',
                          alpha=alpha*snr_alpha,
                          linewidth=linewidth,
                          label='compressional confidence interval')]

    out = Container(artists)
    return out

