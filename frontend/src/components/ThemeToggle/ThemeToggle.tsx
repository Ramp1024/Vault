import type { Theme } from '../../hooks/useTheme'

type ThemeToggleProps = {
    theme: Theme
    onSelect: (theme: Theme) => void
}

const SunIcon = () => (
    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
        <g fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
            <circle cx="12" cy="12" r="3.6" />
            <path d="M12 2.5v2M12 19.5v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2.5 12h2M19.5 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" />
        </g>
    </svg>
)

const MoonIcon = () => (
    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
        <path
            fill="currentColor"
            d="M12.74 2.32a.75.75 0 0 0-.82-.98A9.5 9.5 0 1 0 22.66 12.1a.75.75 0 0 0-.98-.82 7 7 0 0 1-8.94-8.95Z"
        />
    </svg>
)

export function ThemeToggle({ theme, onSelect }: ThemeToggleProps) {
    return (
        <div className="theme-toggle" role="group" aria-label="Theme">
            <button
                type="button"
                className={`theme-toggle-option ${theme === 'light' ? 'active' : ''}`}
                aria-pressed={theme === 'light'}
                aria-label="Light mode"
                onClick={() => onSelect('light')}
            >
                <SunIcon />
            </button>
            <button
                type="button"
                className={`theme-toggle-option ${theme === 'dark' ? 'active' : ''}`}
                aria-pressed={theme === 'dark'}
                aria-label="Dark mode"
                onClick={() => onSelect('dark')}
            >
                <MoonIcon />
            </button>
        </div>
    )
}
