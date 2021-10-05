using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net
{
    public class NetManager : NetModule
    {
        // private List<NetModule> 
        public override Torch Forward(Torch @in)
        {
            return base.Forward(@in);
        }
        public override Torch Backup(Torch @out)
        {
            return base.Backup(@out);
        }
    }
}
