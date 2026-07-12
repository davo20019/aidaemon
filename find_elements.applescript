tell application "System Events"
    tell process "Google Chrome"
        -- Let's find the AXWebArea
        -- Based on the tree, we can traverse to it
        set win to window 1
        set webArea to my find_web_area(win)
        if webArea is missing value then
            return "Could not find AXWebArea"
        end if
        
        -- Let's find all buttons containing "Like"
        set matches to {}
        my find_like_buttons(webArea, matches)
        
        set res to "Found " & (count of matches) & " matches:
"
        repeat with m in matches
            set res to res & m & "
"
        end repeat
        return res
    end tell
end tell

on find_web_area(elem)
    tell application "System Events"
        try
            if role of elem is "AXWebArea" then
                return elem
            end if
        on error
            return missing value
        end try
        
        try
            set children to every UI element of elem
            repeat with child in children
                set found to my find_web_area(child)
                if found is not missing value then
                    return found
                end if
            end repeat
        on error
            -- ignore
        end try
    end tell
    return missing value
end find_web_area

on find_like_buttons(elem, matches)
    tell application "System Events"
        try
            set r to role of elem
            set desc to description of elem
            set t to title of elem
            set h to ""
            try
                set h to help of elem
            end try
            
            set t_str to ""
            if t is not missing value then set t_str to t as string
            set d_str to ""
            if desc is not missing value then set d_str to desc as string
            set h_str to ""
            if h is not missing value then set h_str to h as string
            
            set is_like to (t_str contains "Like") or (d_str contains "Like") or (h_str contains "Like") or (t_str contains "Me gusta") or (d_str contains "Me gusta") or (h_str contains "Me gusta")
            
            if is_like or r is "AXButton" then
                -- let's check if it's a Like button
                if is_like or (d_str contains "like") or (t_str contains "like") then
                    set pos to position of elem
                    set sz to size of elem
                    set end of matches to "Role: " & r & ", Title: '" & t_str & "', Desc: '" & d_str & "', Help: '" & h_str & "', Pos: " & (item 1 of pos) & "," & (item 2 of pos) & ", Size: " & (item 1 of sz) & "x" & (item 2 of sz)
                end if
            end if
        on error err
            -- ignore
        end try
        
        try
            set children to every UI element of elem
            repeat with child in children
                my find_like_buttons(child, matches)
            end repeat
        on error
            -- ignore
        end try
    end tell
end find_like_buttons
