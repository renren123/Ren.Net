using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    public class NetModule
    {
        public virtual Torch Forward(Torch @in) 
        {
            return null;
        }

        public virtual Torch Backup(Torch @out)
        {
            return null;
        }
    }
}
